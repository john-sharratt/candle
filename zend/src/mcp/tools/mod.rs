//! The tools zend's MCP server offers, one file each: its definition — name,
//! description, input schema — and its call. The work tools
//! (`docs/claude_zend_gateway.md` §3.2) delegate to zend and read what came
//! back; the operations tools (§8) see and run the cluster — its members,
//! their commands, the build agents, each machine's service supervisor, the
//! substrate — and every one but `cluster_status` and `cluster_logs` is for
//! Admins only, with each decision gated on the user's approval
//! ([`super::ops`]).
//!
//! A tool's name says what it acts on and what it does to it — `task_start`,
//! `service_status` — never the process that serves it.
//!
//! **The descriptions carry the rules.** claude.ai discards a server's
//! `instructions` (§13.3), so what Claude must do to keep its thread — attach
//! first, name each task for what it is for, read the journal after its
//! context is compacted — is said in the descriptions, the one channel every
//! client delivers. **And they define their terms**: a model meets these tools
//! cold, so a description glosses the system's own words where it uses them —
//! a worker, a member, a build agent, keep — and opens with what the tool does.

use std::sync::Arc;
use std::time::Duration;

use mcp_server::{protocol, Session, Store};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

use super::caller::Caller;
use super::reported::{self, Mark};
use super::state::{ATTACHED, DEFAULT_WAIT_S};
use crate::session::ZendSession;

mod agent_job;
mod agent_job_log;
mod archive;
mod attach;
mod cluster_audit;
mod cluster_command;
mod cluster_job_status;
mod cluster_logs;
mod cluster_status;
mod gateway_command;
mod journal_grep;
mod journal_note;
mod journal_read;
mod layer_read;
mod layer_toggle;
mod nudge;
mod reply;
mod service_control;
mod service_status;
mod service_trial;
mod stop;
mod task;
mod threads;
mod timeline_delete;
mod timeline_read;
pub(crate) mod wait;

/// What a tool answers when it needs the model and zend is still loading it.
pub const LOADING: &str = "the model is still loading; try again shortly";

/// What a tool call runs with.
pub struct Ctx {
    pub session: Arc<ZendSession>,
    pub caller: Caller,
    /// The MCP session the call came on, its id, and where it is kept.
    pub mcp: Session,
    pub mcp_id: String,
    pub store: Arc<dyn Store>,
    /// Where the call says each step of the work it waits on: the call's event
    /// stream, when it is answered on one.
    pub progress: Option<UnboundedSender<String>>,
}

impl Ctx {
    /// The thread the session is attached to, which `"self"` names.
    pub fn attached(&self) -> Option<String> {
        self.mcp.held(ATTACHED).map(str::to_string)
    }
}

/// A tool's answer: its text, whether it is an error the model should read as
/// one, and the journal entries it hands over, to be marked reported once it
/// is delivered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub text: String,
    pub is_error: bool,
    pub marks: Vec<Mark>,
}

impl Answer {
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
            marks: Vec::new(),
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
            marks: Vec::new(),
        }
    }

    pub fn marking(self, marks: Vec<Mark>) -> Self {
        Self { marks, ..self }
    }

    /// Keep this answer's marks for `owner`: it was delivered.
    pub fn delivered(&self, session: &ZendSession, owner: &str) {
        if self.marks.is_empty() {
            return;
        }
        if let Some(records) = session.shared_substrate() {
            reported::keep(&records, owner, &self.marks);
        }
    }

    /// The `tools/call` result: always a text block, which every client reads.
    pub fn result(&self) -> Value {
        protocol::text_result(&self.text, self.is_error)
    }
}

/// Every tool's definition, for `tools/list`.
pub fn definitions() -> Value {
    json!([
        attach::definition(),
        task::definition(),
        wait::definition(),
        reply::definition(),
        nudge::definition(),
        stop::definition(),
        journal_read::definition(),
        journal_grep::definition(),
        journal_note::definition(),
        threads::definition(),
        archive::definition(),
        cluster_status::definition(),
        cluster_logs::definition(),
        cluster_audit::definition(),
        cluster_command::definition(),
        cluster_job_status::definition(),
        gateway_command::definition(),
        agent_job::definition(),
        agent_job_log::definition(),
        service_status::definition(),
        service_trial::definition(),
        service_control::definition(),
        layer_read::definition(),
        layer_toggle::definition(),
        timeline_read::definition(),
        timeline_delete::definition(),
    ])
}

/// Run tool `name` with `args`.
pub async fn call(ctx: &Ctx, name: &str, args: &Value) -> Answer {
    match name {
        attach::NAME => attach::call(ctx, args),
        task::NAME => task::call(ctx, args).await,
        wait::NAME => wait::call(ctx, args).await,
        reply::NAME => reply::call(ctx, args).await,
        nudge::NAME => nudge::call(ctx, args),
        stop::NAME => stop::call(ctx, args).await,
        journal_read::NAME => journal_read::call(ctx, args),
        journal_note::NAME => journal_note::call(ctx, args),
        journal_grep::NAME => journal_grep::call(ctx, args),
        threads::NAME => threads::call(ctx, args),
        archive::NAME => archive::call(ctx, args),
        cluster_status::NAME => cluster_status::call(ctx, args).await,
        cluster_logs::NAME => cluster_logs::call(ctx, args).await,
        cluster_audit::NAME => cluster_audit::call(ctx, args).await,
        cluster_command::NAME => cluster_command::call(ctx, args).await,
        cluster_job_status::NAME => cluster_job_status::call(ctx, args).await,
        gateway_command::NAME => gateway_command::call(ctx, args).await,
        agent_job::NAME => agent_job::call(ctx, args).await,
        agent_job_log::NAME => agent_job_log::call(ctx, args).await,
        service_status::NAME => service_status::call(ctx, args).await,
        service_trial::NAME => service_trial::call(ctx, args).await,
        service_control::NAME => service_control::call(ctx, args).await,
        layer_read::NAME => layer_read::call(ctx, args),
        layer_toggle::NAME => layer_toggle::call(ctx, args).await,
        timeline_read::NAME => timeline_read::call(ctx, args),
        timeline_delete::NAME => timeline_delete::call(ctx, args).await,
        other => Answer::error(format!("there is no tool `{other}`")),
    }
}

/// A string argument, refused when missing or empty.
pub fn string<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("`{key}` is missing: it is a non-empty string"))
}

/// How long a call waits, and what it asked for when its client's ceiling cut
/// that short.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wait {
    pub secs: u64,
    /// The `wait_s` asked for, when it was more than the client allows.
    pub asked: Option<u64>,
}

impl Wait {
    pub fn duration(self) -> Duration {
        Duration::from_secs(self.secs)
    }

    /// What an answer says when the wait was cut: nothing otherwise.
    pub fn note(self) -> Option<String> {
        self.asked.map(|asked| {
            format!(
                "(You asked to wait {asked} s; this client's calls are cut off sooner, so this \
                 one waited {} s. Call again to keep waiting.)",
                self.secs
            )
        })
    }
}

/// The seconds a call may wait: `wait_s` if given, the default otherwise, at
/// most what the session's client allows.
pub fn wait_secs(ctx: &Ctx, args: &Value) -> Wait {
    let ceiling = ctx.mcp.client.wait_ceiling();
    let wanted = args
        .get("wait_s")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_WAIT_S);
    Wait {
        secs: wanted.min(ceiling),
        asked: (wanted > ceiling).then_some(wanted),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_a_name_a_description_and_an_object_schema() {
        let defs = definitions();
        let names: Vec<&str> = defs
            .as_array()
            .unwrap()
            .iter()
            .map(|d| {
                assert!(d["description"].as_str().unwrap().len() > 40, "{d}");
                assert_eq!(d["inputSchema"]["type"], "object", "{d}");
                d["name"].as_str().unwrap()
            })
            .collect();
        assert_eq!(
            names,
            [
                "thread_attach",
                "task_start",
                "task_wait",
                "task_reply",
                "task_nudge",
                "task_stop",
                "journal_read",
                "journal_grep",
                "journal_note",
                "thread_list",
                "thread_archive",
                "cluster_status",
                "cluster_logs",
                "cluster_audit",
                "cluster_command",
                "cluster_job_status",
                "gateway_command",
                "agent_job",
                "agent_job_log",
                "service_status",
                "service_trial",
                "service_control",
                "layer_read",
                "layer_toggle",
                "timeline_read",
                "timeline_delete"
            ]
        );
    }

    /// **A wait its client cut short says so**, with what was asked and what
    /// was waited; one within the ceiling says nothing.
    #[test]
    fn a_wait_cut_short_by_its_client_says_so() {
        let cut = Wait {
            secs: 200,
            asked: Some(600),
        };
        assert_eq!(
            cut.note().unwrap(),
            "(You asked to wait 600 s; this client's calls are cut off sooner, so this one \
             waited 200 s. Call again to keep waiting.)"
        );
        assert_eq!(
            Wait {
                secs: 45,
                asked: None
            }
            .note(),
            None
        );
    }

    #[test]
    fn an_answer_is_a_text_block() {
        assert_eq!(
            Answer::error("no").result().to_string(),
            r#"{"content":[{"type":"text","text":"no"}],"isError":true}"#
        );
    }

    #[test]
    fn a_string_argument_is_trimmed_and_an_empty_one_refused() {
        let args = json!({"a": "  x ", "b": "  "});
        assert_eq!(string(&args, "a").unwrap(), "x");
        assert!(string(&args, "b").is_err());
        assert!(string(&args, "c").is_err());
    }
}
