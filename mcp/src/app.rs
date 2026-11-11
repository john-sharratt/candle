//! The aggregated MCP server (`docs/mcp_app.md` §3): one server card, every
//! backend's tools in one list, and each call run by the backend that lists
//! its tool — its progress relayed as it streams. The protocol, the sessions
//! and the challenge are `mcp_server`'s; nothing here knows what any tool does.
//!
//! Every listing and every call is logged, one line each: which backend, how
//! long it took, and — at `WARN` — why one failed.
//!
//! Beside the backends' tools it serves its own two, `guide_list` and
//! `guide_read` (`crate::guides`), and adds to each backend tool a guide lists
//! the line naming its guides.
//!
//! Every call but `get_user_context` names its conversation by `ref`
//! (`docs/private_mode.md` §3): a call without a live one of the caller's is
//! refused, and the ref goes on to the backend in `_meta`. `get_user_context`
//! begins or goes on with a conversation, and is served with the client's
//! guide in front (`crate::context`); a call that did something is recorded
//! as an episode of its conversation.

use std::collections::{BTreeSet, HashMap};
use std::net::IpAddr;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::http::HeaderMap;
use futures::future::join_all;
use mcp_server::http::now_ms;
use mcp_server::protocol::text_result;
use mcp_server::refs::{Ref, Refs};
use mcp_server::{Call, Identity, NoCaller, SealedStore, Server, Store};
use serde_json::{json, Value};

use crate::backend::{Backend, Link, OnBehalf};
use crate::caller::Caller;
use crate::context;
use crate::conversation::{self, EXPIRED, FORGED, NO_REF, REF};
use crate::guides::tools as guide_tools;
use crate::guides::Library;
use crate::last_listed::LastListed;
use crate::listing::{self, Listing};
use crate::liveness::Seen;
use crate::routes::Routes;
use crate::served::Served;

/// What every request is served with.
pub struct App {
    pub identity: Identity,
    /// Sessions carried whole in their ids: the aggregator holds nothing on
    /// one, so every member reads a session another opened.
    pub store: Arc<SealedStore>,
    pub backends: Vec<Backend>,
    pub link: Link,
    /// Which backend runs each tool, by name, as every listing has said.
    pub routes: Routes,
    /// Each backend's last listing for each caller, listed in its place while
    /// it cannot be.
    pub last_listed: LastListed,
    /// Which listing each session was served, so a call answered on another
    /// tells its client the list changed.
    pub served: Served,
    /// The guides served beside the backends' tools (`docs/mcp_guides.md`).
    pub library: Library,
    /// The backend tools the last listing marked as only reading.
    pub read_only: RwLock<BTreeSet<String>>,
    /// What conversation refs open under: the key zend issues them with.
    pub refs: Refs,
    /// Each ref's last use as this process saw it.
    pub seen: Seen,
    /// Whether this process has listed, and the record tasks' turn to.
    pub listing: Listing,
}

/// Every backend's tools in backend order, and which backend runs each: a
/// name a backend earlier in the list already gave is the earlier one's.
pub fn merge(lists: Vec<(usize, Vec<Value>)>) -> (Vec<Value>, HashMap<String, usize>) {
    let mut tools = Vec::new();
    let mut routes = HashMap::new();
    for (backend, list) in lists {
        for tool in list {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            if routes.contains_key(name) {
                tracing::warn!(
                    tool = name,
                    "mcp: a tool another backend already lists is left out"
                );
                continue;
            }
            routes.insert(name.to_string(), backend);
            tools.push(tool);
        }
    }
    (tools, routes)
}

/// The first line of `text`, at most 200 characters: what a log line quotes.
fn first_line(text: &str) -> String {
    text.lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(200)
        .collect()
}

/// How a call ran: answered, or given up because its client left.
enum Ran {
    Answered(Value),
    Left,
}

/// What listing every backend found: the tools of those that answered, why
/// each of the others did not, and the listing version each answered on.
pub(crate) struct Listed {
    tools: Vec<Value>,
    failed: Vec<String>,
    versions: Vec<Option<String>>,
}

impl App {
    /// Every backend's tools, as `caller` may call them, on behalf of `on`;
    /// the routes learned from them kept.
    pub(crate) async fn list(&self, caller: &Caller, on: &OnBehalf<'_>) -> Listed {
        let listed = join_all(self.backends.iter().map(|b| async {
            let started = Instant::now();
            let got = self.link.tools(b, caller, on).await;
            (got, started.elapsed())
        }))
        .await;
        let who = caller.key();
        let mut lists = Vec::new();
        let mut why_failed = Vec::new();
        let mut versions = vec![None; self.backends.len()];
        for (i, (got, took)) in listed.into_iter().enumerate() {
            let backend = &self.backends[i].name;
            match got {
                Ok((tools, version)) => {
                    self.listing.mark_listed();
                    versions[i] = version;
                    tracing::info!(
                        backend,
                        tools = tools.len(),
                        ms = took.as_millis() as u64,
                        "mcp: listed"
                    );
                    self.last_listed.keep(&who, i, &tools);
                    lists.push((i, tools));
                }
                Err(why) => {
                    let last = self.last_listed.recall(&who, i);
                    tracing::warn!(
                        backend,
                        ms = took.as_millis() as u64,
                        listed_as_last = last.as_ref().map_or(0, Vec::len),
                        "mcp: listing failed: {why}"
                    );
                    why_failed.push(format!("the {backend} tools could not be listed: {why}"));
                    if let Some(tools) = last {
                        lists.push((i, tools));
                    }
                }
            }
        }
        let (tools, learned) = merge(lists);
        context::mark_read_only(
            &mut self.read_only.write().unwrap_or_else(|e| e.into_inner()),
            &tools,
        );
        self.routes.learn(learned);
        Listed {
            tools,
            failed: why_failed,
            versions,
        }
    }

    /// `name` called on `backend` for `caller`: a memory call at the path only
    /// the members serving the memory declare, so the gateway sends it to one
    /// of them ([`Backend::path_for`]). The listing the backend answered on is
    /// noted against what the session was served.
    pub(crate) async fn call_served(
        &self,
        backend: usize,
        caller: &Caller,
        on: &OnBehalf<'_>,
        name: &str,
        args: &Value,
        step: &(dyn Fn(String) + Sync),
    ) -> Value {
        let (answer, version) = self
            .link
            .call(&self.backends[backend], caller, on, name, args, step)
            .await;
        if let Some(version) = version {
            self.served.answered(on.id, backend, &version, now_ms());
        }
        answer
    }

    pub(crate) fn route(&self, name: &str) -> Option<usize> {
        self.routes.get(name)
    }

    /// The session `call` is made on behalf of, in the conversation `named`.
    fn on_behalf<'a>(call: &'a Call<Self>, named: Option<&'a str>) -> OnBehalf<'a> {
        OnBehalf {
            id: &call.session_id,
            session: &call.session,
            conversation_ref: named,
        }
    }

    /// The call, run by the backend that lists `name` — the tools listed
    /// first when none has, as for a tool new to every listing so far — or
    /// answered here when it is a guide tool; in the conversation `named`.
    async fn run(&self, call: &Call<Self>, name: &str, args: &Value, named: Option<&str>) -> Ran {
        if guide_tools::is_guide_tool(name) {
            let answer = guide_tools::call(&self.library, name, args);
            let guide = args.get("name").and_then(Value::as_str).unwrap_or("");
            tracing::info!(tool = name, guide, "mcp: guide served");
            return Ran::Answered(answer);
        }
        let on = Self::on_behalf(call, named);
        let backend = match self.route(name) {
            Some(b) => b,
            None => {
                let listed = self.list(&call.caller, &on).await;
                match self.route(name) {
                    Some(b) => b,
                    None if listed.failed.is_empty() => {
                        return Ran::Answered(text_result(
                            &format!("there is no tool `{name}`"),
                            true,
                        ))
                    }
                    None => return Ran::Answered(text_result(&listed.failed.join("; "), true)),
                }
            }
        };
        let step = |line: String| call.step(line);
        let started = Instant::now();
        let to = &self.backends[backend].name;
        // A client gone drops the backend's call with it, so the backend does
        // not take an answer nobody read as delivered.
        let answer = tokio::select! {
            answer = self.call_served(backend, &call.caller, &on, name, args, &step) => answer,
            _ = call.abandoned() => {
                tracing::info!(tool = name, backend = to, "mcp: the client left before the answer");
                return Ran::Left;
            }
        };
        let ms = started.elapsed().as_millis() as u64;
        if context::is_error(&answer) {
            let text = context::text_of(&answer).unwrap_or("");
            tracing::warn!(
                tool = name,
                backend = to,
                ms,
                "mcp: call answered an error: {}",
                first_line(text)
            );
        } else {
            tracing::info!(tool = name, backend = to, ms, "mcp: called");
        }
        Ran::Answered(answer)
    }

    /// The user's context as served to `client`, from the memory's answer
    /// to `get_user_context`: the client's guide in front of it. `None` when
    /// the memory answered an error, which is logged.
    fn served_context(&self, client: &str, answer: &Value) -> Option<String> {
        let text = context::text_of(answer).unwrap_or("");
        if context::is_error(answer) {
            tracing::warn!(
                client,
                "mcp: the user's context could not be read: {}",
                first_line(text)
            );
            return None;
        }
        Some(context::compose(
            self.library.for_client(client),
            client,
            text,
        ))
    }

    /// Whether the last listing marked `name` as only reading.
    pub(crate) fn reads_only(&self, name: &str) -> bool {
        self.read_only
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(name)
    }

    /// The conversation `named` names for `call`'s caller, live at `now_ms`:
    /// its ref, as written, and what it names — or why the call is refused.
    /// A ref older than this process can vouch for is asked of the record
    /// both aggregators read (`docs/private_mode.md` §3.2).
    pub(crate) async fn conversation(
        this: &Arc<Self>,
        call: &Call<Self>,
        named: Option<&str>,
        now_ms: u64,
    ) -> Result<(String, Ref), String> {
        let text = named.ok_or(NO_REF)?;
        let r = this
            .refs
            .open(&call.caller.key(), text)
            .map_err(|_| FORGED.to_string())?;
        if !this.seen.vouches(&r, now_ms) {
            match Self::ask_used(this, call, text).await {
                Ok(Some(used)) if r.live(Some(used), now_ms) => this.seen.heard(&r, used),
                Ok(_) => return Err(EXPIRED.into()),
                Err(why) => {
                    return Err(format!(
                        "this conversation's `ref` could not be checked ({why}); try the same call \
                         again shortly"
                    ))
                }
            }
        }
        Self::note_use(this, call, text, &r, now_ms);
        Ok((text.to_string(), r))
    }

    /// When the record says the conversation `text` names was last used.
    async fn ask_used(
        this: &Arc<Self>,
        call: &Call<Self>,
        text: &str,
    ) -> Result<Option<u64>, String> {
        let backend = this
            .route(context::GET)
            .ok_or("no backend lists the user's context yet")?;
        let on = Self::on_behalf(call, Some(text));
        let answer = this
            .call_served(
                backend,
                &call.caller,
                &on,
                context::REF_SEEN,
                &json!({}),
                &|_| {},
            )
            .await;
        let said = context::text_of(&answer).unwrap_or("");
        if context::is_error(&answer) {
            return Err(first_line(said));
        }
        let read: Value = serde_json::from_str(said)
            .map_err(|_| format!("the record answered `{}`", first_line(said)))?;
        Ok(read.get("used_ms").and_then(Value::as_u64))
    }

    /// Note the conversation `text` names used at `now_ms`, and tell the
    /// record so in the background when that is due: the call never waits
    /// for it.
    fn note_use(this: &Arc<Self>, call: &Call<Self>, text: &str, r: &Ref, now_ms: u64) {
        if !this.seen.used(r, now_ms) {
            return;
        }
        let Some(backend) = this.route(context::GET) else {
            return;
        };
        let this = Arc::clone(this);
        let caller = call.caller.clone();
        let id = call.session_id.clone();
        let session = call.session.clone();
        let text = text.to_string();
        tokio::spawn(async move {
            let on = OnBehalf {
                id: &id,
                session: &session,
                conversation_ref: Some(&text),
            };
            let answer = this
                .call_served(
                    backend,
                    &caller,
                    &on,
                    context::REF_SEEN,
                    &json!({"used_ms": now_ms}),
                    &|_| {},
                )
                .await;
            if context::is_error(&answer) {
                tracing::warn!(
                    "mcp: a conversation's use could not be told: {}",
                    first_line(context::text_of(&answer).unwrap_or(""))
                );
            }
        });
    }

    /// `get_user_context`: run by the memory's backend, which begins or goes
    /// on with the conversation and answers its ref first; served with the
    /// client's guide in front. A ref it went on with is a use.
    async fn begin(this: &Arc<Self>, call: &Call<Self>, args: &Value) -> Value {
        let Ran::Answered(mut answer) = this.run(call, context::GET, args, None).await else {
            return text_result("the client stopped listening", true);
        };
        let client = call.session.client.name();
        if let Some(served) = this.served_context(client, &answer) {
            context::replace_text(&mut answer, &served);
        }
        if let Some(text) = args.get(REF).and_then(Value::as_str).map(str::trim) {
            if let Ok(r) = this.refs.open(&call.caller.key(), text) {
                Self::note_use(this, call, text, &r, now_ms());
            }
        }
        answer
    }
}

/// Record `episode` — a call that did something for `caller` — on `backend`,
/// the one that keeps the memory. A record refused — the memory's backend
/// restarting, its model still loading — is tried again, every
/// [`context::EPISODE_RETRY`] up to [`context::EPISODE_TRIES`] times, so a
/// picture drawn while zend reloads is still remembered.
pub(crate) async fn record_episode(
    this: &App,
    backend: usize,
    caller: &Caller,
    on: &OnBehalf<'_>,
    episode: &Value,
) {
    let tool = episode["tool"].as_str().unwrap_or("");
    for attempt in 1..=context::EPISODE_TRIES {
        let answer = this
            .call_served(backend, caller, on, context::EPISODE, episode, &|_| {})
            .await;
        if !context::is_error(&answer) {
            return;
        }
        tracing::warn!(
            tool,
            attempt,
            "mcp: an episode could not be recorded: {}",
            first_line(context::text_of(&answer).unwrap_or(""))
        );
        if attempt < context::EPISODE_TRIES {
            tokio::time::sleep(context::EPISODE_RETRY).await;
        }
    }
}

impl Server for App {
    type Caller = Caller;
    type Answer = Value;

    fn identity(&self) -> &Identity {
        &self.identity
    }

    /// Whoever the believed headers name ([`crate::believed`] dropped any
    /// other's before the router). Whether they may use a backend's tools is
    /// that backend's to answer, from the role table it reads.
    fn caller(&self, headers: &HeaderMap, _peer: Option<IpAddr>) -> Result<Caller, NoCaller> {
        Caller::of(headers).ok_or(NoCaller::SignedOut)
    }

    fn owner(&self, caller: &Caller) -> String {
        caller.key()
    }

    fn store(&self) -> Result<Arc<dyn Store>, String> {
        Ok(self.store.clone())
    }

    /// The tools of every backend. A backend down — restarting, still loading
    /// its model — is listed as it last was for this caller (`LastListed`):
    /// a client keeps the list it was given across conversations, so a short
    /// one would outlast the outage. One never listed is left out. Only when
    /// nothing at all can be listed is the list refused, since claude.ai shows
    /// a refused list as no tools at all.
    ///
    /// The guide tools come first, each backend tool a guide lists carries
    /// the line naming its guides, and every tool but `get_user_context`
    /// requires `ref`.
    async fn tools(this: &Arc<Self>, call: &Call<Self>) -> Result<Value, String> {
        let mut listed = this.list(&call.caller, &Self::on_behalf(call, None)).await;
        if listed.tools.is_empty() && !listed.failed.is_empty() {
            return Err(listed.failed.join("; "));
        }
        if listed.failed.is_empty() {
            let served: BTreeSet<&str> = listed
                .tools
                .iter()
                .filter_map(|t| t.get("name").and_then(Value::as_str))
                .collect();
            for (tool, guides) in this.library.unserved(&served) {
                tracing::warn!(
                    tool,
                    guides = guides.join(", "),
                    "mcp: a guide lists a tool no backend serves"
                );
            }
        }
        this.library.decorate(&mut listed.tools);
        let mut tools = guide_tools::definitions(&this.library);
        tools.append(&mut listed.tools);
        conversation::require_ref(&mut tools);
        this.served
            .listed(&call.session_id, listed.versions, now_ms());
        Ok(Value::Array(tools))
    }

    /// The call (`App::run`) in the conversation its `ref` names — refused
    /// without a live one — or `get_user_context`, which begins or goes on
    /// with one (`App::begin`). A call that did something is recorded as an
    /// episode of its conversation (`docs/mcp_user_context.md` §2.3).
    async fn call(this: &Arc<Self>, call: &Call<Self>, name: &str, args: &Value) -> Value {
        if context::is_internal(name) {
            return text_result(&format!("there is no tool `{name}`"), true);
        }
        if name == context::GET {
            return Self::begin(this, call, args).await;
        }
        let (named, args) = conversation::take_ref(args);
        let text = match Self::conversation(this, call, named.as_deref(), now_ms()).await {
            Ok((text, _)) => text,
            Err(why) => return text_result(&why, true),
        };
        let Ran::Answered(answer) = this.run(call, name, &args, Some(&text)).await else {
            // A client that left needs no record of an answer it never read.
            return text_result("the client stopped listening", true);
        };
        // Whether it is an episode is decided after the answer has gone, by a
        // listing when this process has none yet (`crate::listing`).
        if context::may_record(name, context::is_error(&answer)) {
            listing::record_later(this, call, name, &args, text);
        }
        answer
    }

    /// The aggregator's listing is the sum of the backends'; each names its own.
    fn tools_version(&self) -> Option<&str> {
        None
    }

    fn list_changed(&self, call: &Call<Self>) -> bool {
        let owed = self.served.take_owed(&call.session_id);
        if owed {
            tracing::info!(
                client = call.session.client.name(),
                "mcp: told a client its tool list changed"
            );
        }
        owed
    }

    fn result(&self, answer: &Value) -> Value {
        answer.clone()
    }

    /// What a delivered answer changes is the backend's, which saw it go.
    async fn delivered(_this: &Arc<Self>, _call: &Call<Self>, _answer: &Value) {}
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn the_lists_merge_in_backend_order_and_a_repeated_name_is_the_first_ones() {
        let (tools, routes) = merge(vec![
            (
                0,
                vec![
                    json!({"name": "repo_task_start"}),
                    json!({"name": "shared"}),
                ],
            ),
            (
                1,
                vec![
                    json!({"name": "bot_list"}),
                    json!({"name": "shared"}),
                    json!({}),
                ],
            ),
        ]);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["repo_task_start", "shared", "bot_list"]);
        assert_eq!(routes["shared"], 0);
        assert_eq!(routes["bot_list"], 1);
        assert_eq!(routes.len(), 3);
    }
}
