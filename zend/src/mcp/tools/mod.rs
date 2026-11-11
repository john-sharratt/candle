//! The tools zend's MCP server offers, one file each: its definition — name,
//! description, input schema — and its call. The work tools
//! (`docs/claude_zend_gateway.md` §3.2) delegate to zend and read what came
//! back; the operations tools (§8) see and run the cluster — its members,
//! their commands, the build agents, each machine's service supervisor, the
//! substrate — and every one but `cluster_status` and the log tools ([`logs`])
//! is for Admins only, with each decision gated on the user's approval
//! ([`super::ops`]). Whatever runs longer than a call — a task, a build, a
//! deploy, a maintenance job — is a job, watched, read and stopped with the
//! five `job_*` tools ([`job`], `docs/jobs.md`).
//!
//! A tool's name says what it acts on and what it does to it —
//! `repo_task_start`, `service_status` — never the process that serves it.
//! What it acts on is named so Claude can route by the name alone, before it
//! has read a description: `repo_` is work in the user's repositories, which
//! Claude has no other way to reach.
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

use mcp_server::access::{self, Access};
use mcp_server::refs::Ref;
use mcp_server::{protocol, Session, Store};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

use super::attached;
use super::caller::Caller;
use super::reported::{self, Mark};
use super::state::DEFAULT_WAIT_S;
use crate::session::ZendSession;

pub use fit::fit;

mod approval_grant;
mod archive;
mod attach;
pub mod cluster_audit;
mod diary;
mod diary_close;
mod diary_create;
mod diary_delete;
mod diary_list;
mod diary_open;
mod diary_read;
mod diary_rotate;
mod diary_write;
pub mod cluster_command;
pub mod cluster_logs;
pub mod cluster_status;
mod file_download_and_commit;
mod file_edit_and_commit;
mod file_upload_and_commit;
mod files;
mod fit;
pub mod gateway_command;
pub mod group_add;
pub mod group_create;
pub mod group_default;
pub mod group_describe;
mod group_remove;
pub mod job;
mod journal_accept;
pub mod journal_create;
mod journal_edit;
mod journal_focus;
mod journal_forget;
mod journal_grep;
pub mod journal_link;
pub mod journal_note;
mod journal_read;
mod landing;
pub mod layer_read;
pub mod layer_toggle;
pub mod logs;
pub mod mem;
pub mod member_memory;
pub mod member_status;
pub mod note_terms;
mod nudge;
mod ref_seen;
mod reply;
pub mod repo_job;
pub mod service;
pub mod service_control;
mod task;
mod threads;
pub mod timeline_delete;
pub mod timeline_read;
pub mod turn_report;
mod user_context;
mod user_episode;
pub(crate) mod wait;
mod zen_chat_close;
mod zen_chat_read;
mod zen_chat_send;

/// What a tool answers when it needs the model and zend is still loading it.
pub const LOADING: &str = "the model is still loading; try again shortly";

/// What a tool call runs with.
#[derive(Clone)]
pub struct Ctx {
    pub session: Arc<ZendSession>,
    pub caller: Caller,
    /// The MCP session the call came on, its id, and where it is kept.
    pub mcp: Session,
    pub mcp_id: String,
    /// The conversation the call is in: the ref the aggregator passed on,
    /// opened for this caller (`docs/private_mode.md` §3.3). A call reaching
    /// zend directly, or naming a ref that does not open, has none.
    pub conversation: Option<Ref>,
    pub store: Arc<dyn Store>,
    /// Where the call says each step of the work it waits on: the call's event
    /// stream, when it is answered on one.
    pub progress: Option<UnboundedSender<String>>,
}

impl Ctx {
    /// The thread the call's conversation is attached to, which `"self"`
    /// names (`crate::mcp::attached`). A call in no conversation has none.
    pub fn attached(&self) -> Option<String> {
        let conversation = self.conversation?.conversation();
        let records = self.session.shared_substrate()?;
        attached::attached(&records, &self.caller.key(), &conversation)
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
        user_context::definition(),
        attach::definition(),
        task::definition(),
        reply::definition(),
        nudge::definition(),
        files::definition("file_read"),
        files::definition("file_list"),
        files::definition("file_search"),
        file_edit_and_commit::definition(),
        file_download_and_commit::definition(),
        file_upload_and_commit::definition(),
        journal_read::definition(),
        journal_grep::definition(),
        journal_note::definition(),
        journal_focus::definition(),
        journal_edit::definition(),
        journal_forget::definition(),
        journal_create::definition(),
        journal_link::definition(),
        journal_accept::definition(),
        group_create::definition(),
        group_describe::definition(),
        group_add::definition(),
        group_remove::definition(),
        group_default::definition(),
        threads::definition(),
        archive::definition(),
        zen_chat_send::definition(),
        zen_chat_read::definition(),
        zen_chat_close::definition(),
        diary_open::definition(),
        diary_create::definition(),
        diary_list::definition(),
        diary_read::definition(),
        diary_write::definition(),
        diary_delete::definition(),
        diary_rotate::definition(),
        diary_close::definition(),
        job::list::definition(),
        job::status::definition(),
        job::read::definition(),
        job::grep::definition(),
        job::cancel::definition(),
        cluster_status::definition(),
        logs::list::definition(),
        logs::read::definition(),
        logs::grep::definition(),
        cluster_audit::definition(),
        member_status::definition(),
        member_memory::definition(),
        cluster_command::definition(),
        gateway_command::definition(),
        approval_grant::definition(),
        repo_job::start::definition(),
        service::deploy::definition(),
        service::status::definition(),
        service_control::definition(),
        layer_read::definition(),
        layer_toggle::definition(),
        timeline_read::definition(),
        timeline_delete::definition(),
    ])
}

/// A synchronous tool's call, run on the blocking pool: it reads the
/// substrate and may commit a record or a journal write, which waits on the
/// disk, and on the pool that wait holds up none of the calls the runtime is
/// running beside it.
async fn blocking(
    ctx: &Ctx,
    args: &Value,
    tool: impl FnOnce(&Ctx, &Value) -> Answer + Send + 'static,
) -> Answer {
    let (ctx, args) = (ctx.clone(), args.clone());
    tokio::task::spawn_blocking(move || tool(&ctx, &args))
        .await
        .unwrap_or_else(|e| Answer::error(format!("the tool failed: {e}")))
}

/// Whether `name` may run in the call's conversation (`docs/private_mode.md`
/// §4): the aggregator refuses first, and zend holds to the same table for
/// any call that reaches it. A call in no conversation keeps no diary.
fn permitted(ctx: &Ctx, name: &str) -> Result<(), &'static str> {
    match ctx.conversation {
        Some(r) => access::permits(name, r.private),
        None if access::access(name) == Access::Diary => Err(access::DIARIES_PRIVATE),
        None => Ok(()),
    }
}

/// Run tool `name` with `args`.
pub async fn call(ctx: &Ctx, name: &str, args: &Value) -> Answer {
    if let Err(why) = permitted(ctx, name) {
        return Answer::error(why);
    }
    match name {
        user_context::NAME => blocking(ctx, args, user_context::call).await,
        user_episode::NAME => blocking(ctx, args, user_episode::call).await,
        ref_seen::NAME => blocking(ctx, args, ref_seen::call).await,
        journal_forget::NAME => blocking(ctx, args, journal_forget::call).await,
        attach::NAME => blocking(ctx, args, attach::call).await,
        task::NAME => task::call(ctx, args).await,
        reply::NAME => reply::call(ctx, args).await,
        nudge::NAME => blocking(ctx, args, nudge::call).await,
        name if files::NAMES.contains(&name) => {
            let name = name.to_string();
            blocking(ctx, args, move |ctx, args| files::call(ctx, &name, args)).await
        }
        file_edit_and_commit::NAME => blocking(ctx, args, file_edit_and_commit::call).await,
        file_download_and_commit::NAME => blocking(ctx, args, file_download_and_commit::call).await,
        file_upload_and_commit::NAME => blocking(ctx, args, file_upload_and_commit::call).await,
        journal_read::NAME => blocking(ctx, args, journal_read::call).await,
        journal_note::NAME => blocking(ctx, args, journal_note::call).await,
        journal_grep::NAME => blocking(ctx, args, journal_grep::call).await,
        journal_edit::NAME => blocking(ctx, args, journal_edit::call).await,
        journal_focus::NAME => blocking(ctx, args, journal_focus::call).await,
        journal_create::NAME => blocking(ctx, args, journal_create::call).await,
        journal_link::NAME => blocking(ctx, args, journal_link::call).await,
        journal_accept::NAME => blocking(ctx, args, journal_accept::call).await,
        group_create::NAME => blocking(ctx, args, group_create::call).await,
        group_describe::NAME => blocking(ctx, args, group_describe::call).await,
        group_add::NAME => blocking(ctx, args, group_add::call).await,
        group_remove::NAME => blocking(ctx, args, group_remove::call).await,
        group_default::NAME => blocking(ctx, args, group_default::call).await,
        threads::NAME => blocking(ctx, args, threads::call).await,
        archive::NAME => blocking(ctx, args, archive::call).await,
        zen_chat_send::NAME => zen_chat_send::call(ctx, args).await,
        zen_chat_read::NAME => zen_chat_read::call(ctx, args).await,
        zen_chat_close::NAME => blocking(ctx, args, zen_chat_close::call).await,
        diary_open::NAME => blocking(ctx, args, diary_open::call).await,
        diary_create::NAME => blocking(ctx, args, diary_create::call).await,
        diary_list::NAME => blocking(ctx, args, diary_list::call).await,
        diary_read::NAME => blocking(ctx, args, diary_read::call).await,
        diary_write::NAME => blocking(ctx, args, diary_write::call).await,
        diary_delete::NAME => blocking(ctx, args, diary_delete::call).await,
        diary_rotate::NAME => blocking(ctx, args, diary_rotate::call).await,
        diary_close::NAME => blocking(ctx, args, diary_close::call).await,
        job::list::NAME => job::list::call(ctx, args).await,
        job::status::NAME => job::status::call(ctx, args).await,
        job::read::NAME => job::read::call(ctx, args).await,
        job::grep::NAME => job::grep::call(ctx, args).await,
        job::cancel::NAME => job::cancel::call(ctx, args).await,
        cluster_status::NAME => cluster_status::call(ctx, args).await,
        logs::list::NAME => logs::list::call(ctx, args).await,
        logs::read::NAME => logs::read::call(ctx, args).await,
        logs::grep::NAME => logs::grep::call(ctx, args).await,
        cluster_audit::NAME => cluster_audit::call(ctx, args).await,
        member_status::NAME => member_status::call(ctx, args).await,
        member_memory::NAME => member_memory::call(ctx, args).await,
        cluster_command::NAME => cluster_command::call(ctx, args).await,
        gateway_command::NAME => gateway_command::call(ctx, args).await,
        approval_grant::NAME => approval_grant::call(ctx, args).await,
        repo_job::start::NAME => repo_job::start::call(ctx, args).await,
        service::deploy::NAME => service::deploy::call(ctx, args).await,
        service::status::NAME => service::status::call(ctx, args).await,
        service_control::NAME => service_control::call(ctx, args).await,
        layer_read::NAME => blocking(ctx, args, layer_read::call).await,
        layer_toggle::NAME => layer_toggle::call(ctx, args).await,
        timeline_read::NAME => timeline_read::call(ctx, args).await,
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
    use std::collections::BTreeSet;

    use super::*;
    use crate::mcp::npcd_tools;

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
                "get_user_context",
                "thread_attach",
                "repo_task_start",
                "repo_task_reply",
                "repo_task_nudge",
                "file_read",
                "file_list",
                "file_search",
                "file_edit_and_commit",
                "file_download_and_commit",
                "file_upload_and_commit",
                "journal_read",
                "journal_grep",
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
                "thread_list",
                "thread_archive",
                "zen_chat_send",
                "zen_chat_read",
                "zen_chat_close",
                "diary_open",
                "diary_create",
                "diary_list",
                "diary_read",
                "diary_write",
                "diary_delete",
                "diary_rotate",
                "diary_close",
                "job_list",
                "job_status",
                "job_read",
                "job_grep",
                "job_cancel",
                "cluster_status",
                "log_list",
                "log_read",
                "log_grep",
                "cluster_audit",
                "member_status",
                "member_memory",
                "cluster_command",
                "gateway_command",
                "approval_grant",
                "repo_job_start",
                "service_deploy",
                "service_status",
                "service_control",
                "layer_read",
                "layer_toggle",
                "timeline_read",
                "timeline_delete"
            ]
        );
    }

    /// **Every tool a guide names is one the connector serves**: zend's, npcd's
    /// — its bot reads and acts, which zend lists for its own workers, and
    /// `image_upload`, its one tool no worker calls — or the aggregator's own
    /// guide tools. The runtime half is the aggregator's `Library::unserved`
    /// warning; this fails the build instead.
    #[test]
    fn every_tool_a_guide_names_is_served() {
        let library = guides::embedded().unwrap();
        let defs = definitions();
        let mut served: BTreeSet<&str> = defs
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|d| d["name"].as_str())
            .collect();
        served.extend(npcd_tools::READS);
        served.extend(npcd_tools::ACTS);
        served.extend(["image_upload", "guide_list", "guide_read"]);
        let unserved = library.unserved(&served);
        assert!(
            unserved.is_empty(),
            "guides name tools nothing serves: {unserved:?}"
        );
    }

    /// **Exactly the tools that only read say so** (`readOnlyHint`): the
    /// aggregator records an episode for every other call
    /// (`docs/mcp_user_context.md` §2.3).
    #[test]
    fn the_tools_that_only_read_say_so() {
        let defs = definitions();
        let read_only: Vec<&str> = defs
            .as_array()
            .unwrap()
            .iter()
            .filter(|d| d["annotations"]["readOnlyHint"] == true)
            .map(|d| d["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            read_only,
            [
                "get_user_context",
                "file_read",
                "file_list",
                "file_search",
                "journal_read",
                "journal_grep",
                "thread_list",
                "zen_chat_read",
                "job_list",
                "job_status",
                "job_read",
                "job_grep",
                "cluster_status",
                "log_list",
                "log_read",
                "log_grep",
                "cluster_audit",
                "member_status",
                "member_memory",
                "service_status",
                "layer_read",
                "timeline_read"
            ]
        );
    }

    /// **Every tool zend lists is in the one table of what tools do**
    /// (`docs/private_mode.md` §4.3), and so are the two the aggregator calls
    /// it by name: a tool added without a line there fails here.
    #[test]
    fn every_tool_is_classified() {
        assert_eq!(access::unclassified(&definitions()), Vec::<String>::new());
        for internal in [user_episode::NAME, ref_seen::NAME] {
            assert!(access::classified(internal).is_some(), "{internal}");
        }
    }

    /// **The table and `readOnlyHint` agree**: what a listing says only
    /// reads, the table lets a private conversation call, and nothing else.
    #[test]
    fn the_table_reads_what_the_listing_says_only_reads() {
        for d in definitions().as_array().unwrap() {
            let name = d["name"].as_str().unwrap();
            let reads = d["annotations"]["readOnlyHint"] == true;
            assert_eq!(access::access(name) == Access::Read, reads, "{name}");
        }
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
