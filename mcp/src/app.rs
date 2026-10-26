//! The aggregated MCP server (`docs/mcp_app.md` §3): one server card, every
//! backend's tools in one list, and each call run by the backend that lists
//! its tool — its progress relayed as it streams. The protocol, the sessions
//! and the challenge are `mcp_server`'s; nothing here knows what any tool does.
//!
//! Every listing and every call is logged, one line each: which backend, how
//! long it took, and — at `WARN` — why one failed.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::http::HeaderMap;
use futures::future::join_all;
use mcp_server::protocol::text_result;
use mcp_server::{Call, Identity, LocalStore, Server, Store};
use serde_json::Value;

use crate::backend::{Backend, Link, OnBehalf};
use crate::caller::Caller;

/// What every request is served with.
pub struct App {
    pub identity: Identity,
    pub gateways: Vec<IpAddr>,
    pub store: Arc<LocalStore>,
    pub backends: Vec<Backend>,
    pub link: Link,
    /// Which backend runs each tool, by name, as they were last listed.
    pub routes: RwLock<HashMap<String, usize>>,
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
                    lists.push((i, tools));
                }
                Err(why) => {
                    tracing::warn!(
                        backend,
                        ms = took.as_millis() as u64,
                        "mcp: listing failed: {why}"
                    );
                    failed.push(i);
                    why_failed.push(format!("{backend}'s tools could not be listed: {why}"));
                }
            }
        }
        let (tools, learned) = merge(lists);
        let mut routes = self.routes.write().unwrap_or_else(|e| e.into_inner());
        *routes = relearn(&routes, learned, &failed);
        Listed {
            tools,
            failed: why_failed,
        }
    }

    fn route(&self, name: &str) -> Option<usize> {
        self.routes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .copied()
    }
}

impl Server for App {
    type Caller = Caller;
    type Answer = Value;

    fn identity(&self) -> &Identity {
        &self.identity
    }

    fn caller(&self, headers: &HeaderMap, peer: Option<IpAddr>) -> Option<Caller> {
        Caller::of(headers, peer, &self.gateways)
    }

    fn owner(&self, caller: &Caller) -> String {
        caller.key()
    }

    fn store(&self) -> Result<Arc<dyn Store>, String> {
        Ok(self.store.clone())
    }

    /// The tools of every backend that answered. A backend down — restarting,
    /// still loading its model — costs only its own tools: claude.ai shows a
    /// refused list as no tools at all, and lists again for each conversation,
    /// so a short list lasts only until the backend is back. Only when no
    /// backend answers is the list refused.
    async fn tools(this: &Arc<Self>, call: &Call<Self>) -> Result<Value, String> {
        let listed = this.list(call).await;
        if listed.tools.is_empty() && !listed.failed.is_empty() {
            Err(listed.failed.join("; "))
        } else {
            Ok(Value::Array(listed.tools))
        }
    }

    /// The call, run by the backend that lists `name` — the tools listed
    /// again first when none does yet, as after a restart.
    async fn call(this: &Arc<Self>, call: &Call<Self>, name: &str, args: &Value) -> Value {
        let backend = match this.route(name) {
            Some(b) => b,
            None => {
                let listed = this.list(call).await;
                match this.route(name) {
                    Some(b) => b,
                    None if listed.failed.is_empty() => {
                        return text_result(&format!("there is no tool `{name}`"), true)
                    }
                    None => return text_result(&listed.failed.join("; "), true),
                }
            }
        };
        let on = OnBehalf {
            id: &call.session_id,
            session: &call.session,
        };
        let step = |line: String| call.step(line);
        let started = Instant::now();
        let to = &this.backends[backend].name;
        // A client gone drops the backend's call with it, so the backend does
        // not take an answer nobody read as delivered.
        let answer = tokio::select! {
            answer = this.link.call(
                &this.backends[backend],
                &call.caller,
                &on,
                name,
                args,
                &step,
            ) => answer,
            _ = call.abandoned() => {
                tracing::info!(tool = name, backend = to, "mcp: the client left before the answer");
                return text_result("the client stopped listening", true);
            }
        };
        let ms = started.elapsed().as_millis() as u64;
        if answer.get("isError").and_then(Value::as_bool) == Some(true) {
            let text = answer["content"][0]["text"].as_str().unwrap_or("");
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
                vec![json!({"name": "task_start"}), json!({"name": "shared"})],
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
        assert_eq!(names, ["task_start", "shared", "bot_list"]);
        assert_eq!(routes["shared"], 0);
        assert_eq!(routes["bot_list"], 1);
        assert_eq!(routes.len(), 3);
    }

    /// **A backend down loses neither the way to another backend's tools nor
    /// its own**: its routes are kept from before, and the others' relearned.
    #[test]
    fn a_backend_that_did_not_answer_keeps_its_routes() {
        let old = HashMap::from([
            ("task_start".to_string(), 0),
            ("bot_list".to_string(), 1),
            ("task_gone".to_string(), 0),
        ]);
        let learned = HashMap::from([("task_start".to_string(), 0), ("task_new".to_string(), 0)]);
        let routes = relearn(&old, learned, &[1]);
        assert_eq!(
            routes,
            HashMap::from([
                ("task_start".to_string(), 0),
                ("task_new".to_string(), 0),
                ("bot_list".to_string(), 1),
            ])
        );
    }
}
