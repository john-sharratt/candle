//! Whether this process has listed its backends, and the record a call leaves
//! behind it: decided after the call's answer has gone, never before.
//!
//! **Which tools only read is a listing's to say**, and a process may not
//! have listed yet: a client keeps the list it was given across restarts, so
//! its calls arrive at a process whose read-only set is empty. Waiting for a
//! listing before answering held a call's answer for seconds after a restart,
//! and up to a minute with a backend down. So a call answers first; what it
//! leaves to record is handed to a task, which lists when no listing has yet
//! succeeded — one listing at a time, and a task that waited while another
//! listed decides with what that listing found — and only then reads which
//! backend serves the tool and whether it only reads.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use mcp_server::Call;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::app::{record_episode, App};
use crate::backend::OnBehalf;
use crate::context;

/// This process's listing of its backends.
#[derive(Debug, Default)]
pub struct Listing {
    /// Whether any backend has been listed successfully.
    listed: AtomicBool,
    /// How many listings the record tasks have run.
    generation: AtomicU64,
    /// Held across a record task's listing: no request takes it. A tokio
    /// mutex does not poison, so a listing that fails or panics releases it.
    turn: Mutex<()>,
}

impl Listing {
    /// Whether any backend has been listed successfully.
    pub fn listed(&self) -> bool {
        self.listed.load(Ordering::SeqCst)
    }

    /// A backend was listed successfully.
    pub fn mark_listed(&self) {
        self.listed.store(true, Ordering::SeqCst);
    }

    /// Run `list` unless a listing has succeeded, or another ran while this
    /// waited for its turn. Whether `list` ran.
    pub async fn ensure(&self, list: impl Future<Output = ()>) -> bool {
        if self.listed() {
            return false;
        }
        let seen = self.generation.load(Ordering::SeqCst);
        let _turn = self.turn.lock().await;
        if self.listed() || self.generation.load(Ordering::SeqCst) != seen {
            return false;
        }
        list.await;
        self.generation.fetch_add(1, Ordering::SeqCst);
        true
    }
}

/// What a call of `name` with `args` that answered, in the conversation
/// `conversation_ref` names, leaves to record — on that conversation —
/// decided and recorded in the background: the call's answer is never held
/// for it.
pub fn record_later(
    this: &Arc<App>,
    call: &Call<App>,
    name: &str,
    args: &Value,
    conversation_ref: String,
) {
    let this = Arc::clone(this);
    let caller = call.caller.clone();
    let id = call.session_id.clone();
    let session = call.session.clone();
    let name = name.to_string();
    let episode = json!({"tool": name, "summary": context::summary(args)});
    tokio::spawn(async move {
        let on = OnBehalf {
            id: &id,
            session: &session,
            conversation_ref: Some(&conversation_ref),
            quiet: false,
        };
        this.listing
            .ensure(async {
                this.list(&caller, &on).await;
            })
            .await;
        let from_backend = this.route(&name).is_some();
        if !context::records(&name, false, from_backend, this.reads_only(&name)) {
            return;
        }
        let Some(backend) = this.route(context::GET) else {
            return;
        };
        record_episode(&this, backend, &caller, &on, &episode).await;
    });
}

#[cfg(test)]
mod tests;
