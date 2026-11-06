//! The tasks a conversation of Zen's starts and steers
//! (`docs/zen_tool_reach.md` §6.3): a person's dialogue, a chat Claude holds
//! with Zen, or a task one of those started. Tasks nest one level: a task a
//! task started holds none of these tools, so it does its work itself.
//!
//! They run through the operations runner as the conversation's person, with
//! `/mcp`'s own implementation (`workers::start`, `answer`, `nudge`), and
//! answer at once with the task's handle: the conversation waits on it with
//! `job_status`, which reads a task as it reads any job. A task started here
//! begins on its starter's branches and uncommitted changes
//! (`crate::conv_inherit`).

use std::collections::{BTreeSet, HashSet};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::caller::Caller;
use super::chats::naming::id_of;
use super::ops_tools::OpsCall;
use super::records::Records;
use super::roles::WorkerKind;
use super::threads::{self, Effort, Priority};
use super::tools::job::JobHandle;
use super::tools::{string, Answer};
use super::workers;

/// The tools, as `/mcp` names them.
pub const START: &str = "repo_task_start";
pub const REPLY: &str = "repo_task_reply";
pub const NUDGE: &str = "repo_task_nudge";

/// Every tool here: what a task's tool list never holds.
pub const TASK_TOOLS: [&str; 3] = [START, REPLY, NUDGE];

/// What a started task is told it begins on (`conv_inherit`).
pub const BEGINS_ON: &str = " It begins on your branches with your uncommitted changes as your \
     last finished tool round saved them; what it changes stays its own until it commits.";

/// Where a conversation stands in the nesting of tasks, which is one level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// A person's dialogue, or a chat: it starts tasks.
    Conversation,
    /// A task a conversation, Claude or a person started: it starts tasks,
    /// each a level down.
    Task,
    /// A task a task started: it starts none.
    Nested,
}

impl Level {
    pub fn starts_tasks(self) -> bool {
        self != Level::Nested
    }
}

/// Hold out of a round's grammar the task tools a conversation at `level`
/// may not call. `offered` is what the round's mode offers: a tool it does
/// not offer has no path to hold out.
pub fn withhold_by_level(
    tools: &mut BTreeSet<&'static str>,
    offered: &HashSet<String>,
    level: Level,
) {
    if !level.starts_tasks() {
        tools.extend(TASK_TOOLS.into_iter().filter(|t| offered.contains(*t)));
    }
}

/// The level of the conversation `conv_id`: a dialogue (no worker names) and
/// a chat are conversations; a task a task started (`Worker::by_task`), or a
/// record that no longer reads, is nested; any other task is a task.
pub fn level_of(records: &dyn Records, conv_id: &str) -> Level {
    let Some((owner, thread, name)) = threads::conv_names(conv_id) else {
        return Level::Conversation;
    };
    match threads::worker(records, owner, thread, name) {
        Some(w) if w.kind.is_chat() => Level::Conversation,
        Some(w) if !w.by_task => Level::Task,
        _ => Level::Nested,
    }
}

/// A task's handle, `task:<thread>/<name>`.
pub fn task_handle(thread: &str, name: &str) -> String {
    JobHandle::Task {
        thread: thread.to_string(),
        name: name.to_string(),
    }
    .to_string()
}

/// The thread a conversation's tasks go on: a task's own thread; for a chat,
/// one named for it, `chat-<its id>` — a chat's own thread is Zen's side of
/// it, kept apart and disposed of with it, and its name is no slug a task's
/// handle can carry; for a dialogue, `dialogue-<first ten hex digits of its
/// id's sha256>`.
pub fn thread_of(conversation: &str) -> String {
    match threads::conv_names(conversation) {
        Some((_, thread, _)) => match id_of(thread) {
            Some(chat) => format!("chat-{chat}"),
            None => thread.to_string(),
        },
        None => {
            let digest = Sha256::digest(conversation.as_bytes());
            let hex: String = digest[..5].iter().map(|b| format!("{b:02x}")).collect();
            format!("dialogue-{hex}")
        }
    }
}

/// The person a call starts or steers tasks as.
fn person(call: &OpsCall) -> Result<&Caller, Answer> {
    call.caller
        .as_ref()
        .ok_or_else(|| Answer::error("a task works for a person, and this call names none"))
}

/// What a started or steered task answers: its handle, and how to follow it.
fn following(thread: &str, name: &str, what: &str) -> Answer {
    let handle = task_handle(thread, name);
    Answer::ok(format!(
        "{what} {handle}. It runs on its own: `job_status {{jobs: [\"{handle}\"], wait_s: 240}}` \
         waits for it and gives its answer and journal entries; `repo_task_reply` answers a \
         question it asks, and `repo_task_nudge` steers it while it runs."
    ))
}

pub fn start(call: &OpsCall, args: &Value) -> Answer {
    let caller = match person(call) {
        Ok(c) => c,
        Err(a) => return a,
    };
    let (name, prompt) = match (string(args, "name"), string(args, "prompt")) {
        (Ok(n), Ok(p)) => (n, p),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let effort = match Effort::of(args.get("effort").and_then(Value::as_str)) {
        Ok(e) => e,
        Err(e) => return Answer::error(e),
    };
    let kind = match WorkerKind::of(args.get("role").and_then(Value::as_str)) {
        Ok(k) => k,
        Err(e) => return Answer::error(e),
    };
    let Some(records) = call.session.shared_substrate() else {
        return Answer::error("the model is still loading");
    };
    let level = level_of(&records, &call.lineage);
    if !level.starts_tasks() {
        return Answer::error(
            "tasks nest one level, and this one is already a level down: do the work in your \
             own rounds",
        );
    }
    let by_task = level == Level::Task;
    let thread = thread_of(&call.lineage);
    match workers::start(
        &call.session,
        caller,
        &thread,
        name,
        prompt,
        (Priority::Normal, effort, kind),
        by_task,
        Some(&call.lineage),
    ) {
        Ok(w) => {
            let mut started = following(&w.thread, &w.name, "Started");
            started.text.push_str(BEGINS_ON);
            started
        }
        Err(e) => Answer::error(e),
    }
}

/// The thread and name `args.task` names, a task handle.
fn task_of(args: &Value) -> Result<(String, String), String> {
    JobHandle::task(string(args, "task")?)
}

pub fn reply(call: &OpsCall, args: &Value) -> Answer {
    let caller = match person(call) {
        Ok(c) => c,
        Err(a) => return a,
    };
    let ((thread, name), text) = match (task_of(args), string(args, "answer")) {
        (Ok(t), Ok(a)) => (t, a),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    match workers::answer(&call.session, caller, &thread, &name, text) {
        Ok(_) => following(&thread, &name, "Replied to"),
        Err(e) => Answer::error(e),
    }
}

pub fn nudge(call: &OpsCall, args: &Value) -> Answer {
    let caller = match person(call) {
        Ok(c) => c,
        Err(a) => return a,
    };
    let ((thread, name), text) = match (task_of(args), string(args, "message")) {
        (Ok(t), Ok(m)) => (t, m),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    match workers::nudge(&call.session, caller, &thread, &name, text) {
        Ok(_) => Answer::ok(format!(
            "Queued for {}: it reads it when its next tool round ends.",
            task_handle(&thread, &name)
        )),
        Err(e) => Answer::error(e),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::mcp::names;
    use crate::mcp::records::MemRecords;
    use crate::mcp::threads::Worker;

    /// **A task's tasks go on its own thread; a chat's and a dialogue's on one
    /// named for it**, the same name every turn, and a name a handle carries.
    #[test]
    fn a_conversations_tasks_go_on_its_thread() {
        assert_eq!(thread_of("mcp:154797f34c5a6ca6:zen-ops/coder"), "zen-ops");
        let chat = thread_of("mcp:154797f34c5a6ca6:~3aaf283faecca2a7/zen");
        assert_eq!(chat, "chat-3aaf283faecca2a7");
        assert!(names::check(&chat).is_ok(), "{chat}");
        let d = thread_of("1a7f30b0-2e8d-4ba4-ab2e-13cc882fc015");
        assert!(
            d.starts_with("dialogue-") && d.len() == "dialogue-".len() + 10,
            "{d}"
        );
        assert_eq!(d, thread_of("1a7f30b0-2e8d-4ba4-ab2e-13cc882fc015"));
        assert_ne!(d, thread_of("another"));
    }

    /// **Tasks nest one level**: a dialogue and a task at the top hold out
    /// nothing; a nested task holds out the task tools — and only the tools
    /// its mode offers.
    #[test]
    fn a_nested_task_holds_out_the_task_tools() {
        let offered: HashSet<String> = [START, REPLY, NUDGE, "file_read"].map(String::from).into();
        let held = |level: Level, offered: &HashSet<String>| {
            let mut tools = BTreeSet::new();
            withhold_by_level(&mut tools, offered, level);
            tools
        };
        assert!(held(Level::Conversation, &offered).is_empty());
        assert!(held(Level::Task, &offered).is_empty());
        assert_eq!(held(Level::Nested, &offered), BTreeSet::from(TASK_TOOLS));
        let partial = HashSet::from([START.to_string()]);
        assert_eq!(held(Level::Nested, &partial), BTreeSet::from([START]));
    }

    fn worker(name: &str, kind: &str, by_task: bool) -> Worker {
        serde_json::from_value(json!({
            "thread": "t", "name": name, "conv_id": threads::conv_id("o", "t", name),
            "prompt": "p", "state": "running", "role": "admin", "priority": "normal",
            "kind": kind, "started_ms": 0, "turn_ms": 0, "beat_ms": 0, "by_task": by_task,
            "activity": null, "no_entry": null
        }))
        .unwrap()
    }

    /// **A conversation's level is read from its worker's record**: a
    /// dialogue and a chat are conversations, a task started from one is a
    /// task, a task a task started is nested, and so is one whose record no
    /// longer reads.
    #[test]
    fn a_level_is_read_from_the_record() {
        let r = MemRecords::default();
        for w in [
            worker("zen", "chat", false),
            worker("top", "coder", false),
            worker("under", "researcher", true),
        ] {
            threads::put_worker(&r, "o", &w).unwrap();
        }
        let level = |name: &str| level_of(&r, &threads::conv_id("o", "t", name));
        assert_eq!(level_of(&r, "1a7f30b0-2e8d"), Level::Conversation);
        assert_eq!(level("zen"), Level::Conversation);
        assert_eq!(level("top"), Level::Task);
        assert_eq!(level("under"), Level::Nested);
        assert_eq!(level("gone"), Level::Nested);
    }
}
