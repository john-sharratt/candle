//! Parts: what a worker starts under itself with `sub_run`
//! (`docs/claude_zend_gateway.md` §6.1).
//!
//! A part is a worker's conversation of its own, under the calling worker's
//! path, owned like it and running as the role it runs as. The call waits for
//! the part server-side, for as long as it takes — a part runs inside zend, so
//! no client's call limit applies — and answers with the part's journal entry
//! and address, which the worker cites in its own.
//!
//! **Restart-safe by name.** A restart re-dispatches the tool calls of a turn
//! whose results never landed. The re-dispatched `sub_run` finds its part by
//! the parent and the name it asked for: it waits on the part while it runs,
//! and returns its result when it ended during the caller's current turn,
//! rather than starting the work again or failing as a clash with a live
//! sibling. A part from an earlier turn of the same worker is not that call's
//! part, so asking for its name again starts a numbered sibling.
//!
//! **A caller that is not a task starts a task.** A person's dialogue and a
//! chat have `sub_run` withheld from their rounds (`docs/zen_tool_reach.md`
//! §6.3), so the call reaches here only from a task or its part. Should one
//! arrive from a conversation that is not a task anyway, it starts a task on
//! that conversation's thread through `repo_task_start`, run as its person,
//! and answers with the task's handle — never a part, and never a refusal. A
//! task's own call always starts a part, so a task never starts a task.
//!
//! The runner is held by the session's `/mcp` state and handed to every tool
//! context; it reaches the session through a weak handle, bound when the
//! session starts loading, so a context never keeps the session alive.

use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use serde_json::json;
use tokio::runtime::Handle;
use zend_tools::{OpsRunner, SubagentRequest, SubagentResponse, SubagentRunner};

use super::names::{leaf, parent_of, split_numbered};
use super::threads::{self, State, Worker};
use super::tools::wait::result_of;
use super::workers::{self, now_ms};
use super::zen_tasks::{self, starts_of, task_handle};
use crate::session::ZendSession;

/// The tool that starts a part, as the catalog names it: what a conversation
/// that is not a task never holds.
pub const SUB_RUN: &str = "sub_run";

/// How long one wait on a part runs before it reads the record again. A part
/// is waited on to its end; the slice only bounds how stale a read can be.
const WAIT_SLICE: Duration = Duration::from_secs(600);

/// The most names a lineage holds below its thread: a task, its part, and that
/// part's part. Every level waits on the one below it, holding its own turn,
/// so the depth is bounded rather than left to the model.
pub const MAX_DEPTH: usize = 3;

/// How many times one call starts its part afresh after the member running it
/// stopped: once is a restart; a second loss is a failure to report.
const RESTARTS: usize = 1;

/// Whether a part's member stopped running it — read as an error with no end,
/// since nothing settled it ([`Worker::observed`]).
fn lost(part: &Worker) -> bool {
    part.state == State::Error && part.ended_ms.is_none()
}

/// Refuse a part under `parent` when the lineage is already as deep as it may
/// go.
pub fn check_depth(parent: &Worker) -> Result<(), String> {
    if parent.name.split('/').count() >= MAX_DEPTH {
        return Err(format!(
            "{} is a part of a part, and may not start parts of its own: do this work in \
             your own turns",
            parent.path()
        ));
    }
    Ok(())
}

/// The `sub_run` runner of one session.
#[derive(Default)]
pub struct Parts {
    /// The session parts start on, and the runtime its tasks run on: a call
    /// arrives on a tool round's own thread, which may be no runtime's.
    bound: OnceLock<(Weak<ZendSession>, Handle)>,
    /// Held while a call finds or starts its part.
    claiming: Mutex<()>,
    /// What starts a task for a caller that is not one: the runner of the
    /// conversations' own `repo_task_start`.
    tasks: OnceLock<Arc<dyn OpsRunner>>,
}

impl Parts {
    /// Bind the runner to the session it starts parts on, and the runtime
    /// that session's tasks are spawned on.
    pub fn bind(&self, session: Weak<ZendSession>, runtime: Handle) {
        let _ = self.bound.set((session, runtime));
    }

    /// Start a caller's tasks through `ops`, the runner its `repo_task_start`
    /// runs on.
    pub fn starting_tasks_with(&self, ops: Arc<dyn OpsRunner>) {
        let _ = self.tasks.set(ops);
    }

    /// Start `req` as a task on `caller`'s thread, for a caller that is not a
    /// task.
    fn start_task(&self, caller: &str, req: SubagentRequest) -> Result<SubagentResponse, String> {
        let ops = self
            .tasks
            .get()
            .ok_or("no runner starts this conversation's tasks")?;
        let args = json!({"name": req.name, "prompt": req.prompt});
        let result = ops.run(caller, zen_tasks::START, args)?;
        Ok(SubagentResponse {
            path: task_handle(&zen_tasks::thread_of(caller), &req.name),
            state: "started".into(),
            result,
        })
    }
}

/// What a call asking for `wanted` under `parent` finds already there: the
/// part of the caller's current turn by that name, when there is one.
///
/// A part is this call's when its leaf's base name is `wanted`, it hangs
/// directly under `parent`, and it started no earlier than the parent's
/// current turn.
pub fn existing<'a>(parent: &Worker, wanted: &str, under: &'a [Worker]) -> Option<&'a Worker> {
    under
        .iter()
        .filter(|w| parent_of(&w.name) == parent.name)
        .filter(|w| split_numbered(leaf(&w.name)).0 == wanted)
        .filter(|w| w.started_ms >= parent.turn_ms)
        .max_by_key(|w| w.started_ms)
}

/// The response for a part that ended: its final response — the shape its
/// worker asked it to answer in, which an entry only summarises — and every
/// journal entry it wrote.
pub fn response(part: &Worker) -> SubagentResponse {
    let result = result_of(part, None);
    SubagentResponse {
        path: part.path(),
        state: match part.state {
            State::Error => "error",
            _ => "done",
        }
        .to_string(),
        result,
    }
}

impl SubagentRunner for Parts {
    fn run(&self, parent: &str, req: SubagentRequest) -> Result<SubagentResponse, String> {
        let caller = parent;
        let Some((owner, thread, name)) = threads::conv_names(caller) else {
            return self.start_task(caller, req);
        };
        let (session, runtime) = self
            .bound
            .get()
            .and_then(|(s, rt)| Some((s.upgrade()?, rt.clone())))
            .ok_or("the model is not running")?;
        // The part's task is spawned, and its wait driven, on the session's
        // runtime, whatever thread this call came on.
        let _entered = runtime.enter();
        let records = session
            .shared_substrate()
            .ok_or("the model is still loading")?;
        // A chat is a conversation, not a task: it starts tasks.
        if !starts_of(&records, caller).parts {
            return self.start_task(caller, req);
        }
        let parent = threads::worker(&records, owner, thread, name)
            .ok_or_else(|| format!("the calling task `T:{thread}/{name}` has no record"))?;
        check_depth(&parent)?;
        let mut part = {
            // Finding and starting are one step, so two calls for one name in
            // a round — or a call and its re-dispatch — share the part one of
            // them started, rather than the second clashing with it.
            let _claim = self.claiming.lock().unwrap_or_else(|e| e.into_inner());
            let now = now_ms();
            let under: Vec<Worker> = threads::workers(&records, owner, thread)
                .into_iter()
                .map(|w| w.observed(now))
                .collect();
            // A part that failed is this call's answer; one whose member
            // stopped running it is started afresh.
            match existing(&parent, &req.name, &under) {
                Some(found) if !lost(found) => found.clone(),
                _ => workers::start_part(&session, owner, &parent, &req.name, &req.prompt)?,
            }
        };
        let mut restarts = 0;
        loop {
            let wanted = [(part.thread.clone(), part.name.clone())];
            let read =
                runtime.block_on(workers::wait(&session, owner, &wanted, WAIT_SLICE, None))?;
            match read.into_iter().next().and_then(|(_, w)| w) {
                Some(w) if w.state.working() => continue,
                // Found running across a restart, its beat ran out while
                // waited on: nothing will finish it.
                Some(w) if lost(&w) && restarts < RESTARTS => {
                    restarts += 1;
                    let _claim = self.claiming.lock().unwrap_or_else(|e| e.into_inner());
                    part = workers::start_part(&session, owner, &parent, &req.name, &req.prompt)?;
                }
                Some(w) => return Ok(response(&w)),
                None => return Err(format!("the part `{}` lost its record", part.path())),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::threads::{Entry, BEAT_LOST_MS};
    use web::auth::Role;

    fn worker(name: &str, started_ms: u64, state: State) -> Worker {
        Worker {
            thread: "review".into(),
            name: name.into(),
            conv_id: threads::conv_id("o", "review", name),
            prompt: "p".into(),
            state,
            role: Role::Admin,
            person: None,
            priority: Default::default(),
            effort: Default::default(),
            kind: Default::default(),
            started_ms,
            turn_ms: started_ms,
            beat_ms: started_ms,
            ended_ms: None,
            entries_after: None,
            entries: Vec::new(),
            final_text: "ok".into(),
            error: None,
            question: None,
            activity: None,
            no_entry: None,
            stopped: false,
            resumable: false,
            resumes: 0,
            by_task: false,
            audit: None,
            checked: false,
            runner: None,
        }
    }

    /// **A re-dispatched call finds its own part**: the newest part of that
    /// name directly under the caller, started in the caller's current turn —
    /// never a cousin's, and never one from an earlier turn.
    #[test]
    fn a_call_finds_the_part_of_its_own_turn_by_name() {
        let mut parent = worker("verify", 0, State::Running);
        parent.turn_ms = 100;
        let under = vec![
            parent.clone(),
            worker("verify/read", 50, State::Done),
            worker("verify/read.2", 120, State::Running),
            worker("audit/read", 130, State::Running),
            worker("verify/check", 140, State::Running),
        ];
        let found = existing(&parent, "read", &under).unwrap();
        assert_eq!(found.name, "verify/read.2");
        assert!(existing(&parent, "write", &under).is_none());

        parent.turn_ms = 200;
        assert!(
            existing(&parent, "read", &under).is_none(),
            "a part from an earlier turn is not this call's"
        );
    }

    #[test]
    fn a_parts_response_is_its_path_state_and_entry() {
        let mut done = worker("verify/read", 0, State::Done);
        done.entries = vec![Entry {
            id: 2,
            text: "Entry 2 — read the parser".into(),
        }];
        let r = response(&done);
        assert_eq!(r.path, "T:review/verify/read");
        assert_eq!(r.state, "done");
        assert_eq!(
            r.result,
            "task:review/verify/read — done. Its answer:\nok\n\n\
             Journal entries:\n\n\
             T:review/verify/read#2\nEntry 2 — read the parser"
        );
        let mut failed = worker("verify/read", 0, State::Error);
        failed.error = Some("boom".into());
        let r = response(&failed);
        assert_eq!(
            (r.state.as_str(), r.result.as_str()),
            ("error", "task:review/verify/read — error: boom")
        );
    }

    /// **The lineage is bounded**: a task and its part may start parts, a
    /// part's part may not.
    #[test]
    fn a_part_of_a_part_starts_no_parts() {
        assert!(check_depth(&worker("verify", 0, State::Running)).is_ok());
        assert!(check_depth(&worker("verify/read", 0, State::Running)).is_ok());
        let err = check_depth(&worker("verify/read/line", 0, State::Running)).unwrap_err();
        assert!(
            err.starts_with("T:review/verify/read/line is a part of a part"),
            "{err}"
        );
    }

    /// A part whose member stopped running it is lost — an error nothing
    /// ended — and one that failed on its own is not.
    #[test]
    fn a_part_is_lost_only_when_nothing_ended_it() {
        let mut p = worker("verify/read", 0, State::Running);
        p.beat_ms = 0;
        let cut_off = p.clone().observed(BEAT_LOST_MS + 1);
        assert!(lost(&cut_off));
        let mut failed = worker("verify/read", 0, State::Error);
        failed.ended_ms = Some(9);
        assert!(!lost(&failed));
    }

    /// **Tasks nest one level** (`starts_of`): a dialogue and a chat start
    /// tasks; a task at the top starts tasks and parts; a task a task started,
    /// a part, and a task whose record is gone start only parts.
    #[test]
    fn what_each_conversation_may_start() {
        use crate::mcp::records::MemRecords;
        use crate::mcp::roles::WorkerKind;
        use crate::mcp::zen_tasks::Starts;
        let r = MemRecords::default();
        let top = worker("top", 0, State::Running);
        let mut nested = worker("nested", 0, State::Running);
        nested.by_task = true;
        let part = worker("top/read", 0, State::Running);
        let mut chat = worker("chat", 0, State::Running);
        chat.kind = WorkerKind::Chat;
        for w in [&top, &nested, &part, &chat] {
            threads::put_worker(&r, "o", w).unwrap();
        }
        let of = |name: &str| starts_of(&r, &threads::conv_id("o", "review", name));
        assert_eq!(starts_of(&r, "1a7f30b0-dialogue"), Starts::CONVERSATION);
        assert_eq!(of("chat"), Starts::CONVERSATION);
        assert_eq!(of("top"), Starts::task(false));
        assert_eq!(of("nested"), Starts::task(true));
        assert_eq!(of("top/read"), Starts::task(true));
        assert_eq!(of("gone"), Starts::task(true));
    }

    /// Records each call it runs, and answers it.
    #[derive(Default)]
    struct Recording(Mutex<Vec<(String, String, serde_json::Value)>>);

    impl OpsRunner for Recording {
        fn run(&self, caller: &str, tool: &str, args: serde_json::Value) -> Result<String, String> {
            let mut calls = self.0.lock().unwrap();
            calls.push((caller.into(), tool.into(), args));
            Ok("Started".into())
        }
    }

    /// **A conversation that is not a task starts a task, not a part**: its
    /// `sub_run` runs `repo_task_start` as that conversation, on its thread,
    /// and answers with the task's handle. Nothing is bound, so no session is
    /// touched.
    #[test]
    fn a_conversation_that_is_not_a_task_starts_a_task() {
        let parts = Parts::default();
        let ops = Arc::new(Recording::default());
        parts.starting_tasks_with(Arc::clone(&ops) as _);
        let dialogue = "1a7f30b0-2e8d-4ba4-ab2e-13cc882fc015";
        let r = parts
            .run(
                dialogue,
                SubagentRequest {
                    name: "read".into(),
                    prompt: "p".into(),
                },
            )
            .unwrap();
        assert_eq!(r.state, "started");
        assert_eq!(
            r.path,
            format!("task:{}/read", zen_tasks::thread_of(dialogue))
        );
        let calls = ops.0.lock().unwrap();
        assert_eq!(
            *calls,
            [(
                dialogue.to_string(),
                zen_tasks::START.to_string(),
                json!({"name": "read", "prompt": "p"})
            )]
        );
    }
}
