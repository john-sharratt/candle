//! The tasks a conversation of Zen's starts and steers
//! (`docs/zen_tool_reach.md` §6.3): a person's dialogue, or a chat Claude
//! holds with Zen. A task's own tool list never holds these tools, so a task
//! does all of its work itself and never starts another.
//!
//! They run through the operations runner as the conversation's person, with
//! `/mcp`'s own implementation (`workers::start`, `answer`, `nudge`), and
//! answer at once with the task's handle: the conversation waits on it with
//! `job_status`, which reads a task as it reads any job.

use std::collections::{BTreeSet, HashSet};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::caller::Caller;
use super::ops_tools::OpsCall;
use super::parts::SUB_RUN;
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

/// What a conversation may start: parts with `sub_run`, tasks with the task
/// tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Starts {
    pub parts: bool,
    pub tasks: bool,
}

impl Starts {
    /// A person's dialogue, or a chat: tasks, never parts.
    pub const CONVERSATION: Starts = Starts {
        parts: false,
        tasks: true,
    };

    /// A task: parts always, and tasks only when it is not itself `nested` —
    /// started by a task, or a part of one. Tasks nest one level.
    pub fn task(nested: bool) -> Starts {
        Starts {
            parts: true,
            tasks: !nested,
        }
    }
}

/// Hold out of a round's grammar the tools its conversation may not call
/// ([`Starts`]). `offered` is what the round's mode offers: a tool it does not
/// offer has no path to hold out.
pub fn withhold_by_kind(
    tools: &mut BTreeSet<&'static str>,
    offered: &HashSet<String>,
    starts: Starts,
) {
    let mut out: Vec<&'static str> = Vec::new();
    if !starts.parts {
        out.push(SUB_RUN);
    }
    if !starts.tasks {
        out.extend(TASK_TOOLS);
    }
    tools.extend(out.into_iter().filter(|t| offered.contains(*t)));
}

/// What the conversation `conv_id` may start: a dialogue (no worker names)
/// and a chat are conversations; a task a task started (`Worker::by_task`),
/// a part (a name under its task's), or a record that no longer reads, is a
/// task one level down; any other task is a task at the top.
pub fn starts_of(records: &dyn Records, conv_id: &str) -> Starts {
    let Some((owner, thread, name)) = threads::conv_names(conv_id) else {
        return Starts::CONVERSATION;
    };
    match threads::worker(records, owner, thread, name) {
        Some(w) if w.kind.is_chat() => Starts::CONVERSATION,
        Some(w) => Starts::task(w.by_task || name.contains('/')),
        None => Starts::task(true),
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

/// The thread a conversation's tasks go on: a chat's own thread, or one named
/// for a dialogue, `dialogue-<first ten hex digits of its id's sha256>`.
pub fn thread_of(conversation: &str) -> String {
    match threads::conv_names(conversation) {
        Some((_, thread, _)) => thread.to_string(),
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
    let starts = starts_of(&records, &call.lineage);
    if !starts.tasks {
        return Answer::error(
            "tasks nest one level, and this one is already a level down: do the work in your \
             own rounds, or split it with `sub_run`",
        );
    }
    // A conversation starts parts only when it is a task: then what it starts
    // is a level down.
    let by_task = starts.parts;
    let thread = thread_of(&call.lineage);
    match workers::start(
        &call.session,
        caller,
        &thread,
        name,
        prompt,
        (Priority::Normal, effort, kind),
        by_task,
    ) {
        Ok(w) => following(&w.thread, &w.name, "Started"),
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
    use super::*;

    /// **A chat's tasks go on its own thread; a dialogue's on one named for
    /// it**, the same name every turn.
    #[test]
    fn a_conversations_tasks_go_on_its_thread() {
        assert_eq!(thread_of("mcp:154797f34c5a6ca6:zen-ops/zen"), "zen-ops");
        let d = thread_of("1a7f30b0-2e8d-4ba4-ab2e-13cc882fc015");
        assert!(
            d.starts_with("dialogue-") && d.len() == "dialogue-".len() + 10,
            "{d}"
        );
        assert_eq!(d, thread_of("1a7f30b0-2e8d-4ba4-ab2e-13cc882fc015"));
        assert_ne!(d, thread_of("another"));
    }

    /// **Tasks nest one level, and a dialogue never starts a part**: a
    /// dialogue holds out `sub_run`; a task Claude or a person started holds
    /// out nothing; a task a task started, or a part, holds out the task
    /// tools — and only tools its mode offers are held out.
    #[test]
    fn each_kind_holds_out_what_it_may_not_start() {
        let offered: HashSet<String> = [SUB_RUN, START, REPLY, NUDGE, "file_read"]
            .map(String::from)
            .into();
        let held = |starts: Starts, offered: &HashSet<String>| {
            let mut tools = BTreeSet::new();
            withhold_by_kind(&mut tools, offered, starts);
            tools
        };
        assert_eq!(
            held(Starts::CONVERSATION, &offered),
            BTreeSet::from([SUB_RUN])
        );
        assert!(held(Starts::task(false), &offered).is_empty());
        assert_eq!(
            held(Starts::task(true), &offered),
            BTreeSet::from(TASK_TOOLS)
        );
        let bare = HashSet::from(["file_read".to_string()]);
        assert!(held(Starts::CONVERSATION, &bare).is_empty());
    }
}
