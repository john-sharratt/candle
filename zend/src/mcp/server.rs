//! zend's MCP server, at `/v1/mcp` (`docs/mcp_app.md` §3): who it is, who is
//! calling, where its sessions are kept, and its tools — the protocol around
//! them is `mcp_server`'s. Claude reaches it through the aggregator at
//! `code.tokera.com/mcp`, which serves these tools beside npcd's.

use std::net::IpAddr;
use std::sync::{Arc, LazyLock};

use axum::http::HeaderMap;
use mcp_server::{Call, Identity, Server, Store};
use serde_json::Value;

use super::caller::Caller;
use super::state::RecordStore;
use super::tools::{self, Answer, Ctx, LOADING};
use crate::session::ZendSession;

/// How to keep the thread. Claude Code delivers it; claude.ai discards it, so
/// the tools' descriptions say the same.
pub const INSTRUCTIONS: &str = "zend runs work on hardware you own: delegate reading, implementing \
against a written spec, building, testing, reviewing and researching to it, and keep orchestration, \
design and approvals yourself. Attach to the feature's thread with zend_attach before working on it, \
and again after your context is compacted. Start tasks with zend_task — state each in full and name it \
for what it does — and wait on them together with zend_wait. A task's result is its worker's journal \
entry: an entry is evidence, not truth, so check a claim against the turns it cites (journal_read with \
the entry) before acting on it.";

/// What the server is, plainly. The name and the moon are branding borrowed
/// from Zen, the AI of Battle Cities, Tokera's game; the lore lives on the
/// sign-in page a person sees, never in what the model reads, so the server
/// is treated as the tool it is.
pub const DESCRIPTION: &str = "Zen Code: a coding, CI/CD and knowledge/research assistant. \
Delegate work — implementing, testing, building, reviewing, researching — to its workers, which \
run it on zend and keep a journal of what they found. The name is branding from Tokera's game \
Battle Cities; the tools are ordinary engineering tools.";

/// Where the server's icon is served: the Zen moon, from the site's shared
/// brand folder.
pub const ICON_PATH: &str = "/brand/zen.png";

static IDENTITY: LazyLock<Identity> = LazyLock::new(|| Identity {
    name: "zen".into(),
    title: "Zen".into(),
    description: DESCRIPTION.into(),
    instructions: INSTRUCTIONS.into(),
    icon: Some(ICON_PATH.into()),
    version: env!("CARGO_PKG_VERSION").into(),
});

impl Server for ZendSession {
    type Caller = Caller;
    type Answer = Answer;

    fn identity(&self) -> &Identity {
        &IDENTITY
    }

    /// Whoever the gateway names: this member believes the identity headers
    /// only from the gateways it trusts.
    fn caller(&self, headers: &HeaderMap, _peer: Option<IpAddr>) -> Option<Caller> {
        Caller::of(headers, self.roles())
    }

    fn owner(&self, caller: &Caller) -> String {
        caller.key()
    }

    /// The sessions are records in the substrate, so any member answers a
    /// client; there are none to read until the substrate has loaded.
    fn store(&self) -> Result<Arc<dyn Store>, String> {
        self.shared_substrate()
            .map(|s| Arc::new(RecordStore(s)) as Arc<dyn Store>)
            .ok_or_else(|| LOADING.to_string())
    }

    async fn tools(_this: &Arc<Self>, _call: &Call<Self>) -> Result<Value, String> {
        Ok(tools::definitions())
    }

    async fn call(this: &Arc<Self>, call: &Call<Self>, name: &str, args: &Value) -> Answer {
        let ctx = Ctx {
            session: Arc::clone(this),
            caller: call.caller.clone(),
            mcp: call.session.clone(),
            mcp_id: call.session_id.clone(),
            store: Arc::clone(&call.store),
            progress: call.progress.clone(),
        };
        tools::call(&ctx, name, args).await
    }

    fn result(&self, answer: &Answer) -> Value {
        answer.result()
    }

    async fn delivered(this: &Arc<Self>, call: &Call<Self>, answer: &Answer) {
        answer.delivered(this, &call.caller.key());
    }
}
