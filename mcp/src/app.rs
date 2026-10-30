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
//! And it serves the user's context (`crate::context`): the client's guide in
//! front of `get_user_context`, the same context after the first answer of a
//! conversation, and an episode recorded for each call that did something.

use std::collections::{BTreeSet, HashMap};
use std::net::IpAddr;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::http::HeaderMap;
use futures::future::join_all;
use mcp_server::http::now_ms;
use mcp_server::protocol::text_result;
use mcp_server::{Call, Identity, NoCaller, SealedStore, Server, Store};
use serde_json::{json, Value};

use crate::backend::{Backend, Link, OnBehalf};
use crate::caller::Caller;
use crate::context::{self, Conversations};
use crate::guides::tools as guide_tools;
use crate::guides::Library;
use crate::last_listed::LastListed;

/// What every request is served with.
pub struct App {
    pub identity: Identity,
    /// Sessions carried whole in their ids: the aggregator holds nothing on
    /// one, so every member reads a session another opened.
    pub store: Arc<SealedStore>,
    pub backends: Vec<Backend>,
    pub link: Link,
    /// Which backend runs each tool, by name, as they were last listed.
    pub routes: RwLock<HashMap<String, usize>>,
    /// Each backend's last listing for each caller, listed in its place while
    /// it cannot be.
    pub last_listed: LastListed,
    /// The guides served beside the backends' tools (`docs/mcp_guides.md`).
    pub library: Library,
    /// The backend tools the last listing marked as only reading.
    pub read_only: RwLock<BTreeSet<String>>,
    /// When each conversation last had a call.
    pub conversations: Conversations,
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

/// The routes learned from a listing in which the backends in `failed` did
/// not answer: theirs are kept as they were, so one backend down does not
/// lose the way to another's tools, nor its own once it is back.
pub fn relearn(
    old: &HashMap<String, usize>,
    learned: HashMap<String, usize>,
    failed: &[usize],
) -> HashMap<String, usize> {
    let mut routes = learned;
    for (name, backend) in old {
        if failed.contains(backend) && !routes.contains_key(name) {
            routes.insert(name.clone(), *backend);
        }
    }
    routes
}

/// How a call ran: answered, or given up because its client left.
enum Ran {
    Answered(Value),
    Left,
}

/// What listing every backend found: the tools of those that answered, and
/// why each of the others did not.
struct Listed {
    tools: Vec<Value>,
    failed: Vec<String>,
}

impl App {
    /// Every backend's tools, as `call`'s caller may call them; the routes
    /// learned from them kept.
    async fn list(&self, call: &Call<Self>) -> Listed {
        let on = OnBehalf {
            id: &call.session_id,
            session: &call.session,
        };
        let listed = join_all(self.backends.iter().map(|b| async {
            let started = Instant::now();
            let got = self.link.tools(b, &call.caller, &on).await;
            (got, started.elapsed())
        }))
        .await;
        let who = call.caller.key();
        let mut lists = Vec::new();
        let mut failed = Vec::new();
        let mut why_failed = Vec::new();
        for (i, (got, took)) in listed.into_iter().enumerate() {
            let backend = &self.backends[i].name;
            match got {
                Ok(tools) => {
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
                    failed.push(i);
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
        let mut routes = self.routes.write().unwrap_or_else(|e| e.into_inner());
        *routes = relearn(&routes, learned, &failed);
        Listed {
            tools,
            failed: why_failed,
        }
    }

    /// `name` called on `backend` for `caller`, asked again — up to
    /// [`context::MEMORY_TRIES`] calls in all — while the member it reached
    /// answers that it does not serve the memory: the backend's members are
    /// dealt calls one by one, so the next try is likely one that does.
    async fn call_served(
        &self,
        backend: usize,
        caller: &Caller,
        on: &OnBehalf<'_>,
        name: &str,
        args: &Value,
        step: &(dyn Fn(String) + Sync),
    ) -> Value {
        context::until_served(name, || {
            self.link
                .call(&self.backends[backend], caller, on, name, args, step)
        })
        .await
    }

    fn route(&self, name: &str) -> Option<usize> {
        self.routes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .copied()
    }

    /// The call, run by the backend that lists `name` — the tools listed
    /// again first when none does yet, as after a restart — or answered here
    /// when it is a guide tool.
    async fn run(&self, call: &Call<Self>, name: &str, args: &Value) -> Ran {
        if guide_tools::is_guide_tool(name) {
            let answer = guide_tools::call(&self.library, name, args);
            let guide = args.get("name").and_then(Value::as_str).unwrap_or("");
            tracing::info!(tool = name, guide, "mcp: guide served");
            return Ran::Answered(answer);
        }
        let backend = match self.route(name) {
            Some(b) => b,
            None => {
                let listed = self.list(call).await;
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
        let on = OnBehalf {
            id: &call.session_id,
            session: &call.session,
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

    /// The user's context for `call`'s client, read from the backend that
    /// keeps the memory — `None` when it cannot be, which is logged and costs
    /// the call nothing.
    async fn read_context(&self, call: &Call<Self>) -> Option<String> {
        let client = call.session.client.name();
        let Some(backend) = self.route(context::GET) else {
            tracing::warn!(client, "mcp: no backend lists the user's context yet");
            return None;
        };
        let on = OnBehalf {
            id: &call.session_id,
            session: &call.session,
        };
        let answer = self
            .call_served(
                backend,
                &call.caller,
                &on,
                context::GET,
                &json!({}),
                &|_| {},
            )
            .await;
        self.served_context(client, &answer)
    }

    /// Whether the last listing marked `name` as only reading.
    fn reads_only(&self, name: &str) -> bool {
        self.read_only
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(name)
    }
}

/// Record that `name`, called with `args`, did something for `call`'s caller
/// — on the backend that keeps the memory, in the background, so the call's
/// answer is never held for it. A record refused — the memory's backend
/// restarting, its model still loading — is tried again, every
/// [`context::EPISODE_RETRY`] up to [`context::EPISODE_TRIES`] times, so a
/// picture drawn while zend reloads is still remembered.
fn record_episode(this: &Arc<App>, call: &Call<App>, name: &str, args: &Value) {
    let Some(backend) = this.route(context::GET) else {
        return;
    };
    let this = Arc::clone(this);
    let caller = call.caller.clone();
    let id = call.session_id.clone();
    let session = call.session.clone();
    let episode = json!({"tool": name, "summary": context::summary(args)});
    tokio::spawn(async move {
        let on = OnBehalf {
            id: &id,
            session: &session,
        };
        let tool = episode["tool"].as_str().unwrap_or("");
        for attempt in 1..=context::EPISODE_TRIES {
            let answer = this
                .call_served(backend, &caller, &on, context::EPISODE, &episode, &|_| {})
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
    });
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
    /// The guide tools come first, and each backend tool a guide lists carries
    /// the line naming its guides.
    async fn tools(this: &Arc<Self>, call: &Call<Self>) -> Result<Value, String> {
        let mut listed = this.list(call).await;
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
        Ok(Value::Array(tools))
    }

    /// The call (`App::run`), with the user's context: `get_user_context`
    /// gets the client's guide in front, the first call of a conversation
    /// gets the context after its answer, and a call that did something is
    /// recorded as an episode (`docs/mcp_user_context.md` §2.2–§2.5).
    async fn call(this: &Arc<Self>, call: &Call<Self>, name: &str, args: &Value) -> Value {
        if name == context::EPISODE {
            return text_result(&format!("there is no tool `{name}`"), true);
        }
        let client = call.session.client.name();
        let who = context::who(&call.caller.key(), client, &call.session_id);
        let claim = this.conversations.claim(&who, now_ms());
        let begins = claim.begins();
        let around = context::around(name, begins);
        // The context is read beside the call: a first call costs one backend
        // round trip, not two. Before any listing has said which backend
        // keeps it, as after a restart, the call lists first and the context
        // is read after it.
        let (ran, user) = match (around.fetch_context, this.route(context::GET)) {
            (false, _) => (this.run(call, name, args).await, None),
            (true, Some(_)) => tokio::join!(this.run(call, name, args), this.read_context(call)),
            (true, None) => {
                let ran = this.run(call, name, args).await;
                (ran, this.read_context(call).await)
            }
        };
        // A client that left neither had the context nor needs a record of
        // an answer it never read.
        let Ran::Answered(mut answer) = ran else {
            return text_result("the client stopped listening", true);
        };
        if around.compose {
            if let Some(served) = this.served_context(client, &answer) {
                context::replace_text(&mut answer, &served);
            }
        }
        if let Some(user) = &user {
            context::append(&mut answer, &context::first_call(user));
            tracing::info!(
                tool = name,
                client,
                "mcp: a conversation's first call carries its context"
            );
        }
        let after = context::after(
            name,
            begins,
            context::is_error(&answer),
            this.route(name).is_some(),
            this.reads_only(name),
            user.is_some(),
        );
        // A claim not kept is given up as it drops — here, or on any way out
        // above, the client gone included.
        if after.stamp {
            claim.keep(now_ms());
        }
        if after.record {
            record_episode(this, call, name, args);
        }
        answer
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

    /// **A backend down loses neither the way to another backend's tools nor
    /// its own**: its routes are kept from before, and the others' relearned.
    #[test]
    fn a_backend_that_did_not_answer_keeps_its_routes() {
        let old = HashMap::from([
            ("repo_task_start".to_string(), 0),
            ("bot_list".to_string(), 1),
            ("task_gone".to_string(), 0),
        ]);
        let learned = HashMap::from([
            ("repo_task_start".to_string(), 0),
            ("task_new".to_string(), 0),
        ]);
        let routes = relearn(&old, learned, &[1]);
        assert_eq!(
            routes,
            HashMap::from([
                ("repo_task_start".to_string(), 0),
                ("task_new".to_string(), 0),
                ("bot_list".to_string(), 1),
            ])
        );
    }
}
