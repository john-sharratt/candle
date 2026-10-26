//! What an app supplies to serve MCP: who it is, who is calling, where its
//! sessions are kept, and its tools. Everything else — the protocol, the
//! sessions, the sign-in challenge, progress on an event stream — is
//! [`crate::http`]'s.

use std::future::{pending, Future};
use std::net::IpAddr;
use std::sync::Arc;

use axum::http::HeaderMap;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::session::{Session, Store};

/// Who a server says it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub name: String,
    pub title: String,
    /// What a client shows of the server, and what Claude may read of it.
    pub description: String,
    /// How to use the tools together. Claude Code delivers it; claude.ai
    /// discards it, so the tools' descriptions say what must be said.
    pub instructions: String,
    /// Where its icon is served, under the origin a request reached it at.
    pub icon: Option<String>,
    pub version: String,
}

/// One request's call: who made it, on which session, and where the work it
/// waits on says each step.
pub struct Call<S: Server + ?Sized> {
    pub caller: S::Caller,
    /// The request's headers, the caller's identity among them.
    pub headers: HeaderMap,
    pub session_id: String,
    pub session: Session,
    pub store: Arc<dyn Store>,
    /// The call's event stream, when it is answered on one: each line sent is
    /// a progress notification.
    pub progress: Option<UnboundedSender<String>>,
}

impl<S: Server + ?Sized> Call<S> {
    /// Say a step of the work, when anyone is listening.
    pub fn step(&self, line: impl Into<String>) {
        if let Some(p) = &self.progress {
            let _ = p.send(line.into());
        }
    }

    /// Resolves once the client answered on an event stream has gone: what
    /// the call would deliver now reaches nobody. A call answered in its
    /// response is dropped with it instead, so this never resolves for one.
    pub async fn abandoned(&self) {
        match &self.progress {
            Some(p) => p.closed().await,
            None => pending::<()>().await,
        }
    }
}

/// An app's MCP server.
pub trait Server: Send + Sync + 'static {
    /// Who is calling, as the app knows them.
    type Caller: Send + Sync + 'static;
    /// What one of its tools answers.
    type Answer: Send + 'static;

    fn identity(&self) -> &Identity;

    /// Who `headers` say is calling, from `peer` when the app was reached
    /// directly; nobody, and the request is answered with the challenge that
    /// starts a sign-in.
    fn caller(&self, headers: &HeaderMap, peer: Option<IpAddr>) -> Option<Self::Caller>;

    /// The key the caller's sessions are kept under.
    fn owner(&self, caller: &Self::Caller) -> String;

    /// Where sessions are kept, or why the server cannot answer yet.
    fn store(&self) -> Result<Arc<dyn Store>, String>;

    /// Every tool, as `tools/list` carries them, or why they could not be
    /// listed. The server is passed as its `Arc`, so a tool can hand its state
    /// on — to a handler, to work that outlives the call.
    fn tools(
        this: &Arc<Self>,
        call: &Call<Self>,
    ) -> impl Future<Output = Result<Value, String>> + Send;

    /// Run tool `name` with `args`.
    fn call(
        this: &Arc<Self>,
        call: &Call<Self>,
        name: &str,
        args: &Value,
    ) -> impl Future<Output = Self::Answer> + Send;

    /// The `tools/call` result an answer is sent as.
    fn result(&self, answer: &Self::Answer) -> Value;

    /// What an answer changes once it reached the client.
    fn delivered(
        this: &Arc<Self>,
        call: &Call<Self>,
        answer: &Self::Answer,
    ) -> impl Future<Output = ()> + Send;
}
