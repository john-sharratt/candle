//! The operations tools — the jobs, building, deploying, and reading the
//! cluster — as one implementation two surfaces call (`docs/jobs.md` §8): Claude
//! through the MCP connector, and a worker of Zen's through its own tool round
//! ([`OpsTools`], the daemon's [`OpsRunner`]). Each tool runs against an
//! [`OpsCall`] — whom it is for and how it may wait — so neither surface has a
//! copy of its own to drift.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use axum::http::{HeaderName, HeaderValue, Method};
use host::cluster::up::Reply;
use host::commands::class::Class;
use journals::scope::Channel;
use serde_json::Value;
use tokio::runtime::Handle;
use web::auth::Role;
use zend_tools::{run_watch, OpsRunner};

use super::approvals::{self, Approval};
use super::caller::{Caller, Person};
use super::npcd_tools;
use super::ops;
use super::roles::WorkerKind;
use super::threads;
use super::tools::mem::Mem;
use super::tools::service::guard;
use super::tools::{
    cluster_audit, cluster_command, cluster_logs, cluster_status, gateway_command, job,
    journal_create, journal_link, journal_note, layer_read, layer_toggle, member_status, repo_job,
    service, service_control, timeline_delete, timeline_read, Answer, Ctx,
};
use super::workers::now_ms;
use super::zen_tasks;
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
    /// What it is depends on the call — a maintenance command's class, a
    /// service's action, a build's `stage` ([`held_by_call`]) — and it acts
    /// as the person.
    ByCall,
    /// It is a decision, held for a person's approval ([`approvals`]).
    Decides,
    /// It writes to the memory's journals, as the person: any member who
    /// writes in their group, held for approval from a turn that read content
    /// from outside.
    Memory,
    /// It starts or steers a task, as the person: a dialogue's or a chat's,
    /// since a task's own tool list never holds it (`zen_tasks`).
    Starts,
    /// npcd's: reads a bot or a world, or draws a picture, as the person
    /// (`npcd_tools`), which npcd checks itself.
    Bots,
    /// npcd's: manages a bot, as the person — a conversation that may change
    /// things; one that says something others read, from a read-out turn, is
    /// held for the person's approval.
    BotActs,
}

/// The tools a conversation of Zen's reaches through [`OpsRunner`], and what
/// each needs (`docs/zen_tool_reach.md` §6.1).
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
    (repo_job::start::NAME, Needs::ByCall),
    (cluster_command::NAME, Needs::ByCall),
    (service_control::NAME, Needs::ByCall),
    (layer_toggle::NAME, Needs::Changes),
    (gateway_command::NAME, Needs::Decides),
    (timeline_delete::NAME, Needs::Decides),
    (journal_note::NAME, Needs::Memory),
    (journal_create::NAME, Needs::Memory),
    (journal_link::NAME, Needs::Memory),
    (zen_tasks::START, Needs::Starts),
    (zen_tasks::REPLY, Needs::Starts),
    (zen_tasks::NUDGE, Needs::Starts),
    ("bot_list", Needs::Bots),
    ("bot_mission_read", Needs::Bots),
    ("bot_message_read", Needs::Bots),
    ("bot_channel_read", Needs::Bots),
    ("bot_discord_read", Needs::Bots),
    ("bot_thread_read", Needs::Bots),
    ("image_generate", Needs::Bots),
    ("image_edit", Needs::Bots),
    ("bot_mission_give", Needs::BotActs),
    ("bot_mission_step", Needs::BotActs),
    ("bot_mission_cancel", Needs::BotActs),
    ("bot_message_send", Needs::BotActs),
    ("bot_channel_post", Needs::BotActs),
];

/// The Care commands that take a member out of service, which a conversation
/// of Zen's asks a person for rather than runs: one could be the member it is
/// running on.
const OUT_OF_SERVICE: [&str; 3] = ["drain", "resync", "rejoin"];

/// What a call of `tool` with `args` by a conversation `acting` as it does
/// is, beyond what [`unfit`] checks: run as asked (`Ok(None)`), held for a
/// person's approval with why (`Ok(Some(why))`), or refused with why.
fn held_by_call(tool: &str, args: &Value, acting: Acting) -> Result<Option<String>, String> {
    let changes = |what: &str| {
        if acting.may_change() {
            Ok(())
        } else {
            Err(format!("{what} is a coder's: this task changes nothing"))
        }
    };
    match tool {
        repo_job::start::NAME if args.get("stage").is_some() => {
            changes("a build that stages what it builds").map(|()| None)
        }
        cluster_command::NAME => {
            let name = args.get("name").and_then(Value::as_str).unwrap_or("?");
            match cluster_command::class_of_call(args)? {
                Class::Look => Ok(None),
                Class::Care if OUT_OF_SERVICE.contains(&name) => changes(&format!("`{name}`"))
                    .map(|()| Some(format!("`{name}` takes a member out of service"))),
                Class::Care => changes(&format!("`{name}`")).map(|()| None),
                Class::Decide => Ok(Some(format!("`{name}` is a decision"))),
                Class::OnTheMachine => Err(format!("`{name}` is done on the member's machine")),
            }
        }
        service_control::NAME => match args.get("action").and_then(Value::as_str) {
            Some("stop") => Ok(Some("stopping a service is a decision".into())),
            Some(a @ ("restart" | "start")) => changes(&format!("a service's {a}")).map(|()| None),
            _ => Err("`action` is restart, stop or start".into()),
        },
        _ => Ok(None),
    }
}

/// The machine and service a call of `tool` with `args` would stop, when the
/// deploy guard covers it: a deploy, and a service's restart.
fn guarded(tool: &str, args: &Value) -> Option<(String, String)> {
    let host = args.get("host").and_then(Value::as_str)?;
    let service = args.get("service").and_then(Value::as_str)?;
    let stops = match tool {
        service::deploy::NAME => true,
        service_control::NAME => args.get("action").and_then(Value::as_str) == Some("restart"),
        _ => false,
    };
    stops.then(|| (host.to_string(), service.to_ascii_lowercase()))
}

/// Hold `tool` with `args`, asked by `call`'s conversation, for its person's
/// approval: what the conversation is told.
fn hold(call: &OpsCall, tool: &str, args: &Value, why: &str) -> Result<String, String> {
    let owner = call
        .caller
        .as_ref()
        .map(Caller::key)
        .ok_or("a call held for approval needs a person to approve it, and none is named")?;
    let records = call
        .session
        .shared_substrate()
        .ok_or("the model is still loading")?;
    approvals::ask(&records, &owner, &call.lineage, tool, args, why, now_ms())
        .map(|a| a.asked())
        .map_err(|e| {
            format!("`{tool}` needs a person's approval, and it could not be asked for: {e}")
        })
}

/// Run a granted `approval` for `call`: its tool with its arguments as
/// recorded, the user's `confirm` added, as the granter.
pub async fn run_granted(approval: &Approval, call: &OpsCall) -> Answer {
    run(&approval.tool, call, &confirmed(&approval.args)).await
}

/// `args` as a granted approval runs them: with the user's `confirm`, which
/// the call was held for.
pub fn confirmed(args: &Value) -> Value {
    let mut args = args.clone();
    if let Some(map) = args.as_object_mut() {
        map.insert("confirm".into(), Value::String(ops::APPROVED.into()));
    }
    args
}

/// Why a conversation `acting` as it does, run as `role`, for a person or not
/// (`person`), may not run `tool`.
fn unfit(tool: &str, role: Role, acting: Acting, person: bool) -> Option<String> {
    let Some((_, needs)) = WORKER_TOOLS.iter().find(|(name, _)| *name == tool) else {
        return Some(format!("`{tool}` is not an operations tool"));
    };
    // The memory is every member's, not an Admin's: a person writes in their
    // own group, as the person the conversation is with.
    if matches!(
        needs,
        Needs::Memory | Needs::Starts | Needs::Bots | Needs::BotActs
    ) {
        return match (person, role.at_least(Role::User)) {
            (false, _) => Some(format!(
                "`{tool}` acts as the person this conversation is with, and it names none"
            )),
            (true, false) => Some(format!(
                "`{tool}` is a member's, and this conversation runs as `{role}`"
            )),
            (true, true) if *needs == Needs::BotActs && !acting.may_change() => {
                Some(format!("`{tool}` is a coder's: this task changes nothing"))
            }
            (true, true) => None,
        };
    }
    if !role.at_least(Role::Admin) {
        return Some(format!(
            "`{tool}` is an Admin's, and this conversation runs as `{role}`"
        ));
    }
    match needs {
        Needs::Changes if !acting.may_change() => {
            Some(format!("`{tool}` is a coder's: this task changes nothing"))
        }
        Needs::ReadsAsPerson | Needs::ByCall | Needs::Decides if !person => Some(match acting {
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
        cluster_command::NAME => cluster_command::run(call, args).await,
        service_control::NAME => service_control::run(call, args).await,
        layer_toggle::NAME => layer_toggle::run(call, args).await,
        gateway_command::NAME => gateway_command::run(call, args).await,
        timeline_delete::NAME => timeline_delete::run(call, args).await,
        journal_note::NAME | journal_create::NAME | journal_link::NAME => {
            memory(name, call, args).await
        }
        zen_tasks::START => zen_tasks::start(call, args),
        zen_tasks::REPLY => zen_tasks::reply(call, args),
        zen_tasks::NUDGE => zen_tasks::nudge(call, args),
        bot if npcd_tools::READS.contains(&bot) || npcd_tools::ACTS.contains(&bot) => {
            npcd_tools::call(call, bot, args).await
        }
        other => Answer::error(format!("`{other}` is not an operations tool")),
    }
}

/// A memory write for `call`, as its person writes through Zen: on the
/// blocking pool, since a write commits to the journals' repository.
async fn memory(name: &str, call: &OpsCall, args: &Value) -> Answer {
    let Some(caller) = call.caller.clone() else {
        return Answer::error(format!(
            "`{name}` writes as a person, and this call names none"
        ));
    };
    let (session, name, args) = (Arc::clone(&call.session), name.to_string(), args.clone());
    tokio::task::spawn_blocking(move || {
        let m = match Mem::of(&session, &caller, Channel::Zen) {
            Ok(m) => m,
            Err(e) => return Answer::error(e),
        };
        match name.as_str() {
            journal_note::NAME => {
                let (journal, text) = (
                    args.get("journal").and_then(Value::as_str).unwrap_or(""),
                    args.get("text").and_then(Value::as_str).unwrap_or(""),
                );
                journal_note::note_in(&m, journal, text)
            }
            journal_create::NAME => journal_create::create_in(&m, &args),
            _ => journal_link::link_in(&m, &args),
        }
    })
    .await
    .unwrap_or_else(|e| Answer::error(format!("the memory write stopped: {e}")))
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
    /// The conversations whose current turn has read content from outside
    /// (`tool_round::READS_UNTRUSTED`): what they write to memory waits for a
    /// person (`docs/zen_tool_reach.md` §6.2). A turn starts clean.
    read_out: Mutex<HashSet<String>>,
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

    /// Conversation `conv_id` began a turn: it has read nothing from outside
    /// in it yet.
    pub fn turn_began(&self, conv_id: &str) {
        self.read_out
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(conv_id);
    }

    /// Whether conversation `conv_id`'s turn has read content from outside.
    fn is_read_out(&self, conv_id: &str) -> bool {
        self.read_out
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(conv_id)
    }

    /// The account key of whoever's message started dialogue `conv_id`'s
    /// turn: whom its retrieval reads the memory as.
    pub fn dialogue_key(&self, conv_id: &str) -> Option<String> {
        self.dialogue_person(conv_id).map(|p| p.key())
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
    fn reading_untrusted(&self, caller: &str) {
        self.read_out
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(caller.to_string());
    }

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
        let decides = WORKER_TOOLS
            .iter()
            .any(|(name, needs)| *name == tool && *needs == Needs::Decides);
        let writes_memory = WORKER_TOOLS
            .iter()
            .any(|(name, needs)| *name == tool && *needs == Needs::Memory);
        let mut held = if decides {
            Some(format!("`{tool}` is a decision"))
        } else if writes_memory && self.is_read_out(caller) {
            Some(
                "this turn read content from outside, and a memory line written from it is read \
                 by everyone in the group: a person checks it first"
                    .into(),
            )
        } else if npcd_tools::SAYS.contains(&tool) && self.is_read_out(caller) {
            Some(
                "this turn read content from outside, and what it says here others read: a \
                 person checks it first"
                    .into(),
            )
        } else {
            held_by_call(tool, &args, acting)?
        };
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
        // **A step the guard refuses becomes a person's decision**, not a dead
        // end: a deploy or restart of the motherd this conversation runs on,
        // or of an app's only ready member (`tools/service/guard.rs`).
        if held.is_none() {
            if let Some((host, service)) = guarded(tool, &args) {
                if let Err(why) = runtime.block_on(guard::check(&call, &host, &service)) {
                    held = Some(why);
                }
            }
        }
        if let Some(why) = held {
            return Err(hold(&call, tool, &args, &why)?);
        }
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
        assert!(unfit("web_fetch", Role::Admin, CODER, true)
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

    /// **Each call is run, held for a person, or refused, as §6.1's table
    /// says**: a Look runs for anyone; a Care runs for a conversation that may
    /// change things; a Decide, a stop, and a step that takes a member out of
    /// service are held; the machine's own switches are refused.
    #[test]
    fn each_call_runs_is_held_or_is_refused_by_its_class() {
        use serde_json::json;
        let cmd = |name: &str| json!({"member": "a0ac88178cb2b57d", "name": name});
        assert_eq!(
            held_by_call(cluster_command::NAME, &cmd("audit"), REVIEWER),
            Ok(None)
        );
        assert_eq!(
            held_by_call(cluster_command::NAME, &cmd("compact"), CHAT),
            Ok(None)
        );
        assert!(
            held_by_call(cluster_command::NAME, &cmd("compact"), RESEARCHER)
                .unwrap_err()
                .contains("coder")
        );
        assert_eq!(
            held_by_call(cluster_command::NAME, &cmd("drain"), CODER),
            Ok(Some("`drain` takes a member out of service".into()))
        );
        assert_eq!(
            held_by_call(cluster_command::NAME, &cmd("restart"), Acting::Dialogue),
            Ok(Some("`restart` is a decision".into()))
        );
        assert!(held_by_call(cluster_command::NAME, &cmd("offline"), CODER).is_err());
        let control =
            |action: &str| json!({"host": "WhiteWorld", "service": "mcp", "action": action});
        assert_eq!(
            held_by_call(service_control::NAME, &control("restart"), CODER),
            Ok(None)
        );
        assert_eq!(
            held_by_call(service_control::NAME, &control("stop"), CODER),
            Ok(Some("stopping a service is a decision".into()))
        );
        let build =
            json!({"repo": "candle", "branch": "main", "program": "cargo", "args": ["check"]});
        assert_eq!(
            held_by_call(repo_job::start::NAME, &build, REVIEWER),
            Ok(None)
        );
        let mut staged = build.clone();
        staged["stage"] = json!("target/release/motherd.exe");
        assert!(held_by_call(repo_job::start::NAME, &staged, REVIEWER).is_err());
        assert_eq!(
            held_by_call(repo_job::start::NAME, &staged, CODER),
            Ok(None)
        );
    }

    /// **A deploy and a restart are the guard's; a granted call carries the
    /// user's `confirm`.**
    #[test]
    fn the_guard_covers_what_stops_a_service_and_a_grant_confirms() {
        use serde_json::json;
        let restart = json!({"host": "BlackWorld", "service": "motherd", "action": "restart"});
        assert_eq!(
            guarded(service_control::NAME, &restart),
            Some(("BlackWorld".into(), "motherd".into()))
        );
        let start = json!({"host": "BlackWorld", "service": "motherd", "action": "start"});
        assert_eq!(guarded(service_control::NAME, &start), None);
        let deploy = json!({"host": "WhiteWorld", "service": "motherd"});
        assert_eq!(
            guarded(service::deploy::NAME, &deploy),
            Some(("WhiteWorld".into(), "motherd".into()))
        );
        assert_eq!(
            confirmed(&json!({"name": "restart"})),
            json!({"name": "restart", "confirm": ops::APPROVED})
        );
    }

    /// **The memory is a member's, not an Admin's**: a User's conversation
    /// writes it as its person, and one naming no person writes nothing.
    #[test]
    fn a_memory_write_needs_a_member_not_an_admin() {
        for tool in [journal_note::NAME, journal_create::NAME, journal_link::NAME] {
            assert_eq!(unfit(tool, Role::User, Acting::Dialogue, true), None);
            assert_eq!(unfit(tool, Role::Admin, RESEARCHER, true), None);
            assert!(unfit(tool, Role::User, Acting::Dialogue, false)
                .unwrap()
                .contains("names none"));
            assert!(unfit(tool, Role::Guest, Acting::Dialogue, true)
                .unwrap()
                .contains("member's"));
        }
    }

    /// **A turn that reads from outside is read-out until the next begins**,
    /// one conversation at a time.
    #[test]
    fn a_turn_is_read_out_after_it_reads_from_outside_until_the_next() {
        let ops = OpsTools::default();
        assert!(!ops.is_read_out("c1"));
        ops.reading_untrusted("c1");
        assert!(ops.is_read_out("c1"));
        assert!(!ops.is_read_out("c2"));
        ops.turn_began("c1");
        assert!(!ops.is_read_out("c1"));
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
