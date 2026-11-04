//! The operations tools — the jobs, building, deploying, and reading the
//! cluster — as one implementation two surfaces call (`docs/jobs.md` §8): Claude
//! through the MCP connector, and a worker of Zen's through its own tool round
//! ([`OpsTools`], the daemon's [`OpsRunner`]). Each tool runs against an
//! [`OpsCall`] — whom it is for and how it may wait — so neither surface has a
//! copy of its own to drift.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use axum::http::{HeaderName, HeaderValue, Method};
use host::cluster::up::Reply;
use serde_json::Value;
use tokio::runtime::Handle;
use web::auth::Role;
use zend_tools::{run_watch, OpsRunner};

use super::caller::{Caller, Person};
use super::ops;
use super::roles::WorkerKind;
use super::threads;
use super::tools::{
    cluster_audit, cluster_logs, cluster_status, job, layer_read, member_status, repo_job, service,
    timeline_read, Answer, Ctx,
};
use crate::session::ZendSession;

/// The longest a worker's call waits: it has no client to cut it off, and a
/// round that waited longer would hold the conversation for nothing.
const WORKER_WAIT_S: u64 = 300;

/// Who an operations tool runs for, and how it may wait.
#[derive(Clone)]
pub struct OpsCall {
    pub session: Arc<ZendSession>,
    /// The lineage its jobs go under: an MCP caller's key, or a worker's
    /// conversation — what keeps its builds on the agent they warmed.
    pub lineage: String,
    /// The person it acts as, their role read when the call began: an MCP
    /// caller, or the person a worker works for. Every request it sends up
    /// states them, so each member and gateway checks their role itself.
    /// `None` for a worker whose record names no person.
    pub caller: Option<Caller>,
    /// Where each step of a wait is said.
    pub progress: Option<Arc<dyn Fn(String) + Send + Sync>>,
    /// The most seconds a call may wait.
    pub ceiling: u64,
    /// A worker of Zen's: its deploys are guarded (§4).
    pub worker: bool,
}

impl OpsCall {
    /// The identity a request it sends states.
    pub fn headers(&self) -> Vec<(HeaderName, HeaderValue)> {
        self.caller
            .as_ref()
            .map(Caller::headers)
            .unwrap_or_default()
    }

    /// Whether it acts as an Admin.
    pub fn admin(&self) -> bool {
        self.caller.as_ref().is_some_and(Caller::is_admin)
    }

    /// Send `method path` up to the cluster as the caller, to member `node`
    /// when named, reading the answer no longer than `read_for`.
    pub async fn up(
        &self,
        method: Method,
        path: &str,
        node: Option<&str>,
        read_for: Duration,
    ) -> Result<Reply, String> {
        ops::up_as(
            &self.session,
            self.headers(),
            method,
            path,
            node,
            None,
            read_for,
        )
        .await
    }
    /// The call an MCP tool call makes.
    pub fn of(ctx: &Ctx) -> OpsCall {
        let progress = ctx
            .progress
            .clone()
            .map(|tx| -> Arc<dyn Fn(String) + Send + Sync> {
                Arc::new(move |line| {
                    let _ = tx.send(line);
                })
            });
        OpsCall {
            session: Arc::clone(&ctx.session),
            lineage: ctx.caller.key(),
            caller: Some(ctx.caller.clone()),
            progress,
            ceiling: ctx.mcp.client.wait_ceiling(),
            worker: false,
        }
    }

    /// Say `line` on the call's progress.
    pub fn say(&self, line: impl Into<String>) {
        if let Some(say) = &self.progress {
            say(line.into());
        }
    }

    /// The seconds a call waits: `wait_s` if given, `default` otherwise, at
    /// most the ceiling; and what it asked for when the ceiling cut it short.
    pub fn wait_of(&self, args: &Value, default: u64) -> (u64, Option<u64>) {
        let wanted = args
            .get("wait_s")
            .and_then(Value::as_u64)
            .unwrap_or(default);
        (
            wanted.min(self.ceiling),
            (wanted > self.ceiling).then_some(wanted),
        )
    }
}

/// Which conversation is calling: a task or chat Claude delegated through
/// `/mcp`, by its kind, or a person's dialogue with Zen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Acting {
    Worker(WorkerKind),
    Dialogue,
}

impl Acting {
    /// Whether it may change what runs (`docs/zen_tool_reach.md` §6.1): a
    /// coder, a chat — which acts for Claude, with Claude's role — and a
    /// person's dialogue. A researcher and a reviewer read untrusted content
    /// and change nothing.
    fn may_change(self) -> bool {
        matches!(
            self,
            Acting::Dialogue | Acting::Worker(WorkerKind::Coder | WorkerKind::Chat)
        )
    }
}

/// What a worker's tool needs of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Needs {
    /// It only reads.
    Reads,
    /// It changes something: a coder's.
    Changes,
    /// It reads as the person the worker works for: the gateway answers only
    /// an Admin's identity.
    ReadsAsPerson,
}

/// The tools a worker of Zen's reaches through [`OpsRunner`], and what each
/// needs.
const WORKER_TOOLS: &[(&str, Needs)] = &[
    (job::list::NAME, Needs::ReadsAsPerson),
    (job::status::NAME, Needs::ReadsAsPerson),
    (job::read::NAME, Needs::ReadsAsPerson),
    (job::grep::NAME, Needs::ReadsAsPerson),
    (job::cancel::NAME, Needs::Changes),
    (service::deploy::NAME, Needs::Changes),
    (service::status::NAME, Needs::Reads),
    (cluster_status::NAME, Needs::ReadsAsPerson),
    (cluster_logs::NAME, Needs::ReadsAsPerson),
    (cluster_audit::NAME, Needs::ReadsAsPerson),
    (member_status::NAME, Needs::ReadsAsPerson),
    (timeline_read::NAME, Needs::ReadsAsPerson),
    (layer_read::NAME, Needs::ReadsAsPerson),
];

/// Why a conversation `acting` as it does, run as `role`, for a person or not
/// (`person`), may not run `tool`.
fn unfit(tool: &str, role: Role, acting: Acting, person: bool) -> Option<String> {
    let Some((_, needs)) = WORKER_TOOLS.iter().find(|(name, _)| *name == tool) else {
        return Some(format!("`{tool}` is not an operations tool"));
    };
    if !role.at_least(Role::Admin) {
        return Some(format!(
            "`{tool}` is an Admin's, and this conversation runs as `{role}`"
        ));
    }
    match needs {
        Needs::Changes if !acting.may_change() => {
            Some(format!("`{tool}` is a coder's: this task changes nothing"))
        }
        Needs::ReadsAsPerson if !person => Some(match acting {
            Acting::Dialogue => format!(
                "`{tool}` reads the cluster as the person this conversation is with, and no \
                 signed-in person's message started this turn"
            ),
            Acting::Worker(_) => format!(
                "`{tool}` reads the cluster as the person this task works for, and its record \
                 names none: it was started before tasks carried one. Ask Claude to start it again"
            ),
        }),
        _ => None,
    }
}

/// Run operations tool `name` for `call`.
pub async fn run(name: &str, call: &OpsCall, args: &Value) -> Answer {
    match name {
        cluster_status::NAME => cluster_status::run(call, args).await,
        cluster_logs::NAME => cluster_logs::run(call, args).await,
        cluster_audit::NAME => cluster_audit::run(call, args).await,
        member_status::NAME => member_status::run(call, args).await,
        timeline_read::NAME => timeline_read::run(call, args).await,
        layer_read::NAME => layer_read::run(call, args),
        job::list::NAME => job::list::run(call, args).await,
        job::status::NAME => job::status::run(call, args).await,
        job::read::NAME => job::read::run(call, args).await,
        job::grep::NAME => job::grep::run(call, args).await,
        job::cancel::NAME => job::cancel::run(call, args).await,
        repo_job::start::NAME => repo_job::start::run(call, args).await,
        service::deploy::NAME => service::deploy::run(call, args).await,
        service::status::NAME => service::status::run(call, args).await,
        other => Answer::error(format!("`{other}` is not an operations tool")),
    }
}

/// The [`OpsRunner`] of one session: the operations tools of every
/// conversation of Zen's — a delegated task's, a chat's, and a person's
/// dialogue's.
#[derive(Default)]
pub struct OpsTools {
    /// The session it runs for, and the runtime its sockets live on: a call
    /// arrives on a tool round's own thread.
    bound: OnceLock<(Weak<ZendSession>, Handle)>,
    /// Each dialogue's person: whose signed-in message started its latest
    /// turn, by conversation id. A dialogue's tools read the cluster as them.
    /// Held in memory, as the turn and its tool rounds run in this process.
    people: Mutex<HashMap<String, Person>>,
}

impl OpsTools {
    /// Bind the runner to its session and that session's runtime.
    pub fn bind(&self, session: Weak<ZendSession>, runtime: Handle) {
        let _ = self.bound.set((session, runtime));
    }

    /// Note who started dialogue `conv_id`'s turn: `None` for a message no
    /// signed-in person sent, which reads nothing as anyone.
    pub fn note_dialogue_person(&self, conv_id: &str, person: Option<Person>) {
        let mut people = self.people.lock().unwrap_or_else(|e| e.into_inner());
        match person {
            Some(p) => {
                people.insert(conv_id.to_string(), p);
            }
            None => {
                people.remove(conv_id);
            }
        }
    }

    /// Whose message started dialogue `conv_id`'s turn.
    fn dialogue_person(&self, conv_id: &str) -> Option<Person> {
        self.people
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(conv_id)
            .cloned()
    }
}

impl OpsRunner for OpsTools {
    fn run(&self, caller: &str, tool: &str, args: Value) -> Result<String, String> {
        let (session, runtime) = self
            .bound
            .get()
            .and_then(|(s, rt)| Some((s.upgrade()?, rt.clone())))
            .ok_or("the model is not running")?;
        let (acting, person, role) = match threads::conv_names(caller) {
            Some((owner, thread, name)) => {
                let records = session
                    .shared_substrate()
                    .ok_or("the model is still loading")?;
                let worker = threads::worker(&records, owner, thread, name)
                    .ok_or_else(|| format!("the calling task `T:{thread}/{name}` has no record"))?;
                (Acting::Worker(worker.kind), worker.person, worker.role)
            }
            None => {
                let person = self.dialogue_person(caller);
                let role = person
                    .as_ref()
                    .map_or(Role::Unauthenticated, |p| p.role(session.roles()));
                (Acting::Dialogue, person, role)
            }
        };
        if let Some(why) = unfit(tool, role, acting, person.is_some()) {
            return Err(why);
        }
        let progress = run_watch::current().map(|watch| -> Arc<dyn Fn(String) + Send + Sync> {
            Arc::new(move |line| watch(line))
        });
        let call = OpsCall {
            session,
            lineage: caller.to_string(),
            caller: person.map(|person| Caller { person, role }),
            progress,
            ceiling: WORKER_WAIT_S,
            worker: true,
        };
        let answer = runtime.block_on(run(tool, &call, &args));
        if answer.is_error {
            Err(answer.text)
        } else {
            Ok(answer.text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A worker reads as an Admin, changes as an Admin's coder, and reads
    /// the cluster as the person it works for**; a tool outside the set is
    /// refused by name.
    const CODER: Acting = Acting::Worker(WorkerKind::Coder);
    const REVIEWER: Acting = Acting::Worker(WorkerKind::Reviewer);
    const RESEARCHER: Acting = Acting::Worker(WorkerKind::Researcher);
    const CHAT: Acting = Acting::Worker(WorkerKind::Chat);

    #[test]
    fn a_worker_reads_as_an_admin_and_changes_as_a_coder() {
        assert_eq!(
            unfit(service::status::NAME, Role::Admin, REVIEWER, false),
            None
        );
        assert_eq!(
            unfit(service::deploy::NAME, Role::Admin, CODER, false),
            None
        );
        assert!(unfit(service::deploy::NAME, Role::Admin, RESEARCHER, true)
            .unwrap()
            .contains("coder"));
        assert!(unfit(job::read::NAME, Role::User, CODER, true)
            .unwrap()
            .contains("Admin"));
        assert_eq!(
            unfit(cluster_status::NAME, Role::Admin, REVIEWER, true),
            None
        );
        assert!(unfit(cluster_logs::NAME, Role::Admin, CODER, false)
            .unwrap()
            .contains("names none"));
        assert!(unfit(job::cancel::NAME, Role::Admin, REVIEWER, true)
            .unwrap()
            .contains("coder"));
        assert!(unfit("repo_job_start", Role::Admin, CODER, true)
            .unwrap()
            .contains("not an operations tool"));
    }

    /// **A chat acts as a coder does, and so does an Admin's dialogue**
    /// (`docs/zen_tool_reach.md` §6.1): both deploy and cancel. A dialogue
    /// reads as the person whose message started its turn, and with no such
    /// person it reads nothing as anyone.
    #[test]
    fn a_chat_and_a_dialogue_act_as_a_coder() {
        for acting in [CHAT, Acting::Dialogue] {
            assert_eq!(
                unfit(service::deploy::NAME, Role::Admin, acting, true),
                None
            );
            assert_eq!(unfit(job::cancel::NAME, Role::Admin, acting, true), None);
            assert_eq!(unfit(member_status::NAME, Role::Admin, acting, true), None);
            assert_eq!(unfit(layer_read::NAME, Role::Admin, acting, true), None);
        }
        assert!(
            unfit(cluster_status::NAME, Role::Admin, Acting::Dialogue, false)
                .unwrap()
                .contains("no signed-in person")
        );
        assert!(
            unfit(cluster_status::NAME, Role::User, Acting::Dialogue, true)
                .unwrap()
                .contains("Admin")
        );
    }

    /// **A dialogue's person is the one whose message started its latest
    /// turn**; a turn no signed-in person started forgets the last one.
    #[test]
    fn a_dialogue_remembers_the_person_of_its_turn() {
        let ops = OpsTools::default();
        let person = Person {
            provider: "google".into(),
            sub: "boss".into(),
            email: String::new(),
            name: String::new(),
        };
        ops.note_dialogue_person("c1", Some(person.clone()));
        assert_eq!(ops.dialogue_person("c1"), Some(person));
        assert_eq!(ops.dialogue_person("c2"), None);
        ops.note_dialogue_person("c1", None);
        assert_eq!(ops.dialogue_person("c1"), None);
    }
}
